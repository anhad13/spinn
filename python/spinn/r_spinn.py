import copy

import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from spinn.util.blocks import Embed, Linear, MLP
from spinn.util.blocks import the_gpu, to_gpu, lstm, bundle, unbundle
from spinn.util.misc import Example, Vocab
from spinn.util.blocks import HeKaimingInitializer
from spinn.util.catalan import ShiftProbabilities
from spinn.util.blocks import LayerNormalization
from spinn.spinn_core_model import SPINN
from spinn.spinn_core_model import BaseModel as SpinnBaseModel
import torch.nn.functional as F


from spinn.data import T_SHIFT, T_REDUCE, T_SKIP



def build_model(data_manager, initial_embeddings, vocab_size,
                num_classes, FLAGS, context_args, composition_args, **kwargs):
    model_cls = BaseModel
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA

    return model_cls(
        model_dim=FLAGS.model_dim,
        word_embedding_dim=FLAGS.word_embedding_dim,
        vocab_size=vocab_size,
        initial_embeddings=initial_embeddings,
        num_classes=num_classes,
        embedding_keep_rate=FLAGS.embedding_keep_rate,
        tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim,
        transition_weight=FLAGS.transition_weight,
        use_sentence_pair=use_sentence_pair,
        lateral_tracking=FLAGS.lateral_tracking,
        tracking_ln=FLAGS.tracking_ln,
        use_tracking_in_composition=FLAGS.use_tracking_in_composition,
        predict_use_cell=FLAGS.predict_use_cell,
        use_difference_feature=FLAGS.use_difference_feature,
        use_product_feature=FLAGS.use_product_feature,
        classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
        mlp_dim=FLAGS.mlp_dim,
        num_mlp_layers=FLAGS.num_mlp_layers,
        mlp_ln=FLAGS.mlp_ln,
        context_args=context_args,
        composition_args=composition_args,
        detach=FLAGS.transition_detach,
        evolution=FLAGS.evolution,
    )
##keep the tracker as optional in RL SPINN??
class Tracker(nn.Module):
    '''The tracker keeps a summary of the parsing process so far. This is the
    "tracking LSTM" as described in the paper.'''

    def __init__(
            self,
            size,
            tracker_size,
            lateral_tracking=True,
            tracking_ln=True):
        '''Args:
            size: input size (parser hidden state) = FLAGS.model_dim
            tracker_size: FLAGS.tracking_lstm_hidden_dim
            (see FLAGS for the rest)'''
        super(Tracker, self).__init__()

        # Initialize layers.
        if lateral_tracking:
            self.buf = Linear()(size, 4 * tracker_size, bias=True)
            self.stack1 = Linear()(size, 4 * tracker_size, bias=False)
            self.stack2 = Linear()(size, 4 * tracker_size, bias=False)
            self.lateral = Linear(initializer=HeKaimingInitializer)(
                tracker_size, 4 * tracker_size, bias=False)
            self.state_size = tracker_size
        else:
            self.state_size = size * 3

        if tracking_ln:
            self.buf_ln = LayerNormalization(size)
            self.stack1_ln = LayerNormalization(size)
            self.stack2_ln = LayerNormalization(size)

        self.lateral_tracking = lateral_tracking
        self.tracking_ln = tracking_ln

        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def forward(self, top_buf, top_stack_1, top_stack_2):
        if self.tracking_ln:
            top_buf = self.buf_ln(top_buf)
            top_stack_1 = self.stack1_ln(top_stack_1)
            top_stack_2 = self.stack2_ln(top_stack_2)

        if self.lateral_tracking:
            tracker_inp = self.buf(top_buf)
            tracker_inp += self.stack1(top_stack_1)
            tracker_inp += self.stack2(top_stack_2)

            batch_size = tracker_inp.size(0)

            if self.h is not None:
                tracker_inp += self.lateral(self.h)
            if self.c is None:
                self.c = to_gpu(Variable(torch.from_numpy(
                    np.zeros((batch_size, self.state_size),
                             dtype=np.float32)),
                    volatile=tracker_inp.volatile))

            # Run tracking lstm.
            self.c, self.h = lstm(self.c, tracker_inp)

            return self.h, self.c
        else:
            return torch.cat([top_buf, top_stack_1, top_stack_2], 1), None

    @property
    def states(self):
        return unbundle((self.c, self.h))

    @states.setter
    def states(self, state_iter):
        if state_iter is not None:
            state = bundle(state_iter)
            self.c, self.h = state.c, state.h

class RLAction(nn.Module):
    '''This NN is used to make a decision(shift/reduce).'''

    def __init__(
            self,
            size,
            out_dim,
            relu_size):
        # Initialize layersi.
	super(RLAction, self).__init__()
        self.relu_size=100
        self.tracker_l = Linear()(size, out_dim, bias=False)
        self.buf_l = Linear()(size, out_dim, bias=False)
        self.stack1_l = Linear()(size, out_dim, bias=False)
        self.stack2_l = Linear()(size, out_dim, bias=False)
        self.ll_after= Linear()(out_dim*3, self.relu_size, bias=True)
        self.post_relu= Linear()(self.relu_size, 2,  bias=True)

    def forward(self, top_buf, top_stack_1, top_stack_2, tracker_h):
        t_tracker=self.tracker_l(tracker_h)
        top_buf = self.buf_l(top_buf)
        top_stack_1 = self.stack1_l(top_stack_1)
        top_stack_2 = self.stack2_l(top_stack_2)
        next_inp=torch.cat([top_buf, top_stack_1, top_stack_2],1)
        out_linear=self.ll_after(next_inp)
        out_relu = F.relu(out_linear)
        out_linear2= self.post_relu(out_relu)
        return out_linear2


class RSPINN(SPINN):

    def __init__(self, args, vocab, predict_use_cell):
        super(RSPINN, self).__init__(args, vocab, predict_use_cell)

        # Optional debug mode.
        self.debug = False
        self.detach = args.detach
        self.evolution = args.evolution

        self.transition_weight = args.transition_weight

        self.wrap_items = args.wrap_items
        self.extract_h = args.extract_h

        # Reduce function for semantic composition.
        self.reduce = args.composition
        if args.tracker_size is not None or args.use_internal_parser:
            self.tracker = Tracker(
                args.size,
                args.tracker_size,
                lateral_tracking=args.lateral_tracking,
                tracking_ln=args.tracking_ln)
            if args.transition_weight is not None:
                # TODO: Might be interesting to try a different network here.
                self.predict_use_cell = predict_use_cell
                if self.tracker.lateral_tracking:
                    tinp_size = self.tracker.state_size * \
                        2 if predict_use_cell else self.tracker.state_size
                else:
                    tinp_size = self.tracker.state_size
                self.transition_net = nn.Linear(tinp_size, 2)
        self.rl_action=RLAction(args.size, 232, 100)

        self.choices = np.array([T_SHIFT, T_REDUCE], dtype=np.int32)

        self.shift_probabilities = ShiftProbabilities()

    def reset_state(self):
        self.memories = []

    def forward(
            self,
            example,
            use_internal_parser=False,
            validate_transitions=True):
        self.n_tokens = (
            example.tokens.data != 0).long().sum(
            1, keepdim=False).tolist()

        if self.debug:
            seq_length = example.tokens.size(1)
            assert all(buf_n <= (seq_length + 1) // 2 for buf_n in self.n_tokens), \
                "All sentences (including cropped) must be the appropriate length."

        self.bufs = example.bufs

        # Notes on adding zeros to bufs/stacks.
        # - After the buffer is consumed, we need one zero on the buffer
        #   used as input to the tracker.
        # - For the first two steps, the stack would be empty, but we add
        #   zeros so that the tracker still gets input.
        zeros = self.zeros = to_gpu(Variable(torch.from_numpy(
            np.zeros(self.bufs[0][0].size(), dtype=np.float32)),
            volatile=self.bufs[0][0].volatile))

        # Initialize Buffers. Trim unused tokens.
        self.bufs = [[zeros] + b[-b_n:]
                     for b, b_n in zip(self.bufs, self.n_tokens)]

        # Initialize Stacks.
        self.stacks = [[zeros, zeros] for buf in self.bufs]

        # Initialize other.
        self.n_reduces = np.zeros(len(self.bufs), dtype=np.int32)
        self.n_steps = np.zeros(len(self.bufs), dtype=np.int32)

        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        if not hasattr(example, 'transitions'):
            # TODO: Support no transitions. In the meantime, must at least pass
            # dummy transitions.
            raise ValueError('Transitions must be included.')
        return self.run(example.transitions,
                        run_internal_parser=True,
                        use_internal_parser=use_internal_parser,
                        validate_transitions=validate_transitions)

    def validate(self, transitions, preds, stacks, bufs, zero_padded=True):
        # Note: There is one zero added to bufs, and two zeros added to stacks.
        # Make sure to adjust for this if using lengths of either.
        buf_adjust = 1 if zero_padded else 0
        stack_adjust = 2 if zero_padded else 0

        _transitions = np.array(transitions)
        _preds = preds.copy()
        _invalid = np.zeros(preds.shape, dtype=np.bool)

        cant_skip = _transitions != T_SKIP
        must_skip = _transitions == T_SKIP

        # Fixup predicted skips.
        if len(self.choices) > 2:
            raise NotImplementedError(
                "Can only validate actions for 2 choices right now.")

        buf_lens = [len(buf) - buf_adjust for buf in bufs]
        stack_lens = [len(stack) - stack_adjust for stack in stacks]

        # Cannot reduce on too small a stack
        must_shift = np.array([length < 2 for length in stack_lens])
        check_mask = np.logical_and(cant_skip, must_shift)
        _invalid += np.logical_and(_preds != T_SHIFT, check_mask)
        _preds[must_shift] = T_SHIFT

        # Cannot shift on too small buf
        must_reduce = np.array([length < 1 for length in buf_lens])
        check_mask = np.logical_and(cant_skip, must_reduce)
        _invalid += np.logical_and(_preds != T_REDUCE, check_mask)
        _preds[must_reduce] = T_REDUCE

        # If the given action is skip, then must skip.
        _preds[must_skip] = T_SKIP

        return _preds, _invalid

    def predict_actions(self, transition_output):
        transition_logdist = F.log_softmax(transition_output)
        transition_preds = transition_logdist.data.cpu().numpy().argmax(axis=1)
        return transition_logdist, transition_preds

    def get_transitions_per_example(self, style="preds"):
        if style == "preds":
            source = "t_preds"
        elif style == "given":
            source = "t_given"
        else:
            raise NotImplementedError

        t_preds = np.concatenate([m['t_preds']
                                  for m in self.memories if 't_preds' in m])
        t_preds = torch.from_numpy(t_preds).long()
        t_logprobs = torch.cat(
            [m['t_logprobs'] for m in self.memories if 't_logprobs' in m], 0).data.cpu()
        t_logprobs = torch.cat(
            [t_logprobs, torch.zeros(t_logprobs.size(0), 1)], 1)
        t_strength = torch.gather(t_logprobs, 1, t_preds.view(-1, 1))

        _transitions = [m[source].reshape(
            1, -1) for m in self.memories if m.get(source, None) is not None]
        transitions = np.concatenate(_transitions).T

        t_strength = torch.exp(t_strength.view(
            *list(reversed(transitions.shape))).t())

        skip_mask = (torch.from_numpy(transitions) == T_SKIP).byte()
        t_strength[skip_mask] = 0.

        return transitions, t_strength

    def t_shift(self, buf, stack, tracking, buf_tops, trackings):
        """SHIFT: Should dequeue buffer and item to stack."""
        buf_tops.append(buf.pop() if len(buf) > 0 else self.zeros)
        trackings.append(tracking)

    def t_reduce(self, buf, stack, tracking, lefts, rights, trackings):
        """REDUCE: Should compose top two items of the stack into new item."""

        # The right-most input will be popped first.
        for reduce_inp in [rights, lefts]:
            if len(stack) > 0:
                reduce_inp.append(stack.pop())
            else:
                if self.debug:
                    raise IndexError
                # If we try to Reduce, but there are less than 2 items on the stack,
                # then treat any available item as the right input, and use zeros
                # for any other inputs.
                # NOTE: Only happens on cropped data.
                reduce_inp.append(self.zeros)

        trackings.append(tracking)

    def t_skip(self):
        """SKIP: Acts as padding and is a noop."""

    def shift_phase(self, tops, trackings, stacks):
        """SHIFT: Should dequeue buffer and item to stack."""
        if len(stacks) > 0:
            shift_candidates = iter(tops)
            for stack in stacks:
                new_stack_item = next(shift_candidates)
                stack.append(new_stack_item)

    def reduce_phase(self, lefts, rights, trackings, stacks):
        if len(stacks) > 0:
            reduced = iter(self.reduce(
                lefts, rights, trackings))
            for stack in stacks:
                new_stack_item = next(reduced)
                stack.append(new_stack_item)

    def reduce_phase_hook(self, lefts, rights, trackings, reduce_stacks):
        pass

    def loss_phase_hook(self):
        pass

    def evolution_params(self):
        """
        The parameters trained by evolution strategy
        """
        return [(k, v) for k, v in zip(self.transition_net.state_dict(
        ).keys(), self.transition_net.state_dict().values())]

    def run(self, inp_transitions, run_internal_parser=False,
            use_internal_parser=False, validate_transitions=True):
        transition_loss = None
        transition_acc = 0.0
        num_transitions = inp_transitions.shape[1]
        batch_size = inp_transitions.shape[0]
        invalid_count = np.zeros(batch_size)

        # Transition Loop
        # ===============

        for t_step in range(num_transitions):
            transitions = inp_transitions[:, t_step]
            transition_arr = list(transitions)

            # A mask based on SKIP transitions.
            cant_skip = np.array(transitions) != T_SKIP
            must_skip = np.array(transitions) == T_SKIP

            # Memories
            # ========
            # Keep track of key values to determine accuracy and loss.
            self.memory = {}

            # Prepare tracker input.
            if self.debug and any(len(buf) < 1 or len(stack)
                                  for buf, stack in zip(self.bufs, self.stacks)):
                # To elaborate on this exception, when cropping examples it is possible
                # that your first 1 or 2 actions is a reduce action. It is unclear if this
                # is a bug in cropping or a bug in how we think about cropping. In the meantime,
                # turn on the truncate batch flag, and set the eval_seq_length
                # very high.
                raise IndexError(
                    "Warning: You are probably trying to encode examples"
                    "with cropped transitions. Although, this is a reasonable"
                    "feature, when predicting/validating transitions, you"
                    "probably will not get the behavior that you expect. Disable"
                    "this exception if you dare.")
            self.memory['top_buf'] = self.wrap_items(
                [buf[-1] if len(buf) > 0 else self.zeros for buf in self.bufs])
            self.memory['top_stack_1'] = self.wrap_items(
                [stack[-1] if len(stack) > 0 else self.zeros for stack in self.stacks])
            self.memory['top_stack_2'] = self.wrap_items(
                [stack[-2] if len(stack) > 1 else self.zeros for stack in self.stacks])

            # Run if:
            # A. We have a tracking component and,
            # B. There is at least one transition that will not be skipped.
            if sum(cant_skip) > 0:#and hasattr(self, 'tracker'):
                tracker_h, tracker_c = self.tracker(
                    self.extract_h(self.memory['top_buf']),
                    self.extract_h(self.memory['top_stack_1']),
                    self.extract_h(self.memory['top_stack_2']))
                out_rl= self.rl_action(
                    tracker_h,
                    self.extract_h(self.memory['top_buf']),
                    self.extract_h(self.memory['top_stack_1']),
                    self.extract_h(self.memory['top_stack_2']))

                # Get hidden output from the tracker. Used to predict
                # transitions.


                # if hasattr(self, 'transition_net'):
                #     transition_inp = [tracker_h]
                #     if self.tracker.lateral_tracking and self.predict_use_cell:
                #         transition_inp += [tracker_c]
                #     if self.detach or self.evolution:
                #         transition_inp = torch.cat(transition_inp, 1).detach()
                #     else:
                #         transition_inp = torch.cat(transition_inp, 1)
                #
                #     transition_output = self.transition_net(transition_inp)

                if run_internal_parser:

                    # Predict Actions
                    # ===============
                    #RSPINN will use the same transition net.
                    # TODO: Mask before predicting. This should simplify things and reduce computation.
                    # The downside is that in the Action Phase, need to be smarter about which stacks/bufs
                    # are selected.
                    transition_logdist, transition_preds = self.predict_actions(
                        out_rl)

                    # Distribution of transitions use to calculate transition
                    # loss.
                    self.memory["t_logprobs"] = transition_logdist

                    # Given transitions.
                    self.memory["t_given"] = transitions

                    # Constrain to valid actions
                    # ==========================

                    validated_preds, invalid_mask = self.validate(
                        transition_arr, transition_preds, self.stacks, self.bufs)
                    if validate_transitions:
                        transition_preds = validated_preds

                    # Keep track of which predictions have been valid.
                    self.memory["t_valid_mask"] = np.logical_not(invalid_mask)
                    invalid_count += invalid_mask

                    # If the given action is skip, then must skip.
                    transition_preds[must_skip] = T_SKIP

                    # Actual transition predictions. Used to measure transition
                    # accuracy.
                    self.memory["t_preds"] = transition_preds

                    # Binary mask of examples that have a transition.
                    self.memory["t_mask"] = cant_skip

                    # If this FLAG is set, then use the predicted actions
                    # rather than the given.
                    if use_internal_parser:
                        transition_arr = transition_preds.tolist()

            # Pre-Action Phase
            # ================

            # TODO: See if PyTorch's 'Advanced Indexing for Tensors and Variables' features would simplify this.

            # For SHIFT
            s_stacks, s_tops, s_trackings, s_idxs = [], [], [], []

            # For REDUCE
            r_stacks, r_lefts, r_rights, r_trackings = [], [], [], []

            batch = zip(transition_arr, self.bufs, self.stacks, self.tracker.states if hasattr(
                self, 'tracker') and self.tracker.h is not None else itertools.repeat(None))

            for batch_idx, (transition, buf, stack,
                            tracking) in enumerate(batch):
                if transition == T_SHIFT:  # shift
                    self.t_shift(buf, stack, tracking, s_tops, s_trackings)
                    s_idxs.append(batch_idx)
                    s_stacks.append(stack)
                elif transition == T_REDUCE:  # reduce
                    self.t_reduce(
                        buf,
                        stack,
                        tracking,
                        r_lefts,
                        r_rights,
                        r_trackings)
                    r_stacks.append(stack)
                elif transition == T_SKIP:  # skip
                    self.t_skip()

            # Action Phase
            # ============

            self.shift_phase(s_tops, s_trackings, s_stacks)
            self.reduce_phase(r_lefts, r_rights, r_trackings, r_stacks)
            self.reduce_phase_hook(r_lefts, r_rights, r_trackings, r_stacks)

            # Memory Phase
            # ============

            # APPEND ALL MEMORIES. MASK LATER.

            self.memories.append(self.memory)

            # Update number of reduces seen so far.
            self.n_reduces += (np.array(transition_arr) == T_REDUCE)

            # Update number of non-skip actions seen so far.
            self.n_steps += (np.array(transition_arr) != T_SKIP)

        # Loss Phase
        # ==========

        if hasattr(self, 'tracker') and hasattr(self, 'transition_net'):
            t_preds = np.concatenate([m['t_preds']
                                      for m in self.memories if 't_preds' in m])
            t_given = np.concatenate([m['t_given']
                                      for m in self.memories if 't_given' in m])
            t_mask = np.concatenate([m['t_mask']
                                     for m in self.memories if 't_mask' in m])
            t_logprobs = torch.cat([m['t_logprobs']
                                    for m in self.memories if 't_logprobs' in m], 0)

            # We compute accuracy and loss after all transitions have complete,
            # since examples can have different lengths when not using skips.

            # Transition Accuracy.
            n = t_mask.shape[0]
            n_skips = n - t_mask.sum()
            n_total = n - n_skips
            n_correct = (t_preds == t_given).sum() - n_skips
            transition_acc = n_correct / float(n_total)

            # Transition Loss.
            index = to_gpu(
                Variable(
                    torch.from_numpy(
                        np.arange(
                            t_mask.shape[0])[t_mask])).long())
            select_t_given = to_gpu(Variable(torch.from_numpy(
                t_given[t_mask]), volatile=not self.training).long())
            select_t_logprobs = torch.index_select(t_logprobs, 0, index)
            transition_loss = nn.NLLLoss()(select_t_logprobs, select_t_given) * \
                self.transition_weight

            self.n_invalid = (invalid_count > 0).sum()
            self.invalid = self.n_invalid / float(batch_size)

        self.loss_phase_hook()

        if self.debug:
            assert all(len(stack) == 3 for stack in self.stacks), \
                "Stacks should be fully reduced and have 3 elements: " \
                "two zeros and the sentence encoding."
            assert all(len(buf) == 1 for buf in self.bufs), \
                "Stacks should be fully shifted and have 1 zero."
	print(self.rl_action.post_relu.weight)
        return [stack[-1]
                for stack in self.stacks], transition_acc, transition_loss
class BaseModel(SpinnBaseModel):

    optimize_transition_loss = True
    def build_rspinn(self, args, vocab, predict_use_cell):
        return RSPINN(args, vocab, predict_use_cell)
    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 embedding_keep_rate=None,
                 tracking_lstm_hidden_dim=4,
                 transition_weight=None,
                 encode_reverse=None,
                 encode_bidirectional=None,
                 encode_num_layers=None,
                 lateral_tracking=None,
                 tracking_ln=None,
                 use_tracking_in_composition=None,
                 predict_use_cell=None,
                 use_sentence_pair=False,
                 use_difference_feature=False,
                 use_product_feature=False,
                 mlp_dim=None,
                 num_mlp_layers=None,
                 mlp_ln=None,
                 classifier_keep_rate=None,
                 context_args=None,
                 composition_args=None,
                 detach=None,
                 evolution=None,
                 **kwargs
                 ):
        super(SpinnBaseModel, self).__init__()

        assert not (
            use_tracking_in_composition and not lateral_tracking), "Lateral tracking must be on to use tracking in composition."

        self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature

        self.hidden_dim = composition_args.size
        self.wrap_items = composition_args.wrap_items
        self.extract_h = composition_args.extract_h

        self.initial_embeddings = initial_embeddings
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim

        classifier_dropout_rate = 1. - classifier_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        # Build parsing component.
        self.spinn = self.build_rspinn(
            composition_args, vocab, predict_use_cell)

        # Build classiifer.
        features_dim = self.get_features_dim()#same as spinn
        self.mlp = MLP(features_dim, mlp_dim, num_classes,
                       num_mlp_layers, mlp_ln, classifier_dropout_rate)

        self.embedding_dropout_rate = 1. - embedding_keep_rate

        # Create dynamic embedding layer.
        self.embed = Embed(
            word_embedding_dim,
            vocab.size,
            vectors=vocab.vectors)

        self.input_dim = context_args.input_dim

        self.encode = context_args.encoder
        self.reshape_input = context_args.reshape_input
        self.reshape_context = context_args.reshape_context

        self.inverted_vocabulary = None


    def build_rspinn(self, args, vocab, predict_use_cell):
        return RSPINN(args, vocab, predict_use_cell)

    def run_spinn(
            self,
            example,
            use_internal_parser,
            validate_transitions=True):
        self.spinn.reset_state()
        h_list, transition_acc, transition_loss = self.spinn(
            example, use_internal_parser=use_internal_parser, validate_transitions=validate_transitions)
        h = self.wrap(h_list)
        return h, transition_acc, transition_loss

    def mc_reinforce(self, rewards, baseline):
        t_preds = np.concatenate([m['t_preds']
                                  for m in self.spinn.memories if 't_preds' in m])
        t_mask = np.concatenate([m['t_mask']
                                 for m in self.spinn.memories if 't_mask' in m])
        t_valid_mask = np.concatenate(
            [m['t_valid_mask'] for m in self.spinn.memories if 't_mask' in m])
        t_logprobs = torch.cat(
            [m['t_logprobs'] for m in self.spinn.memories if 't_logprobs' in m], 0)

	if self.use_sentence_pair:
            # Handles the case of SNLI where each reward is used for two
            # sentences.
            rewards = torch.cat([rewards, rewards], 0)
            baseline = torch.cat([baseline, baseline], 0)

	#print(t_logprobs.shape)
	t_logprobs=t_logprobs.view(1,-1)
	#p_actions = to_gpu(Variable(torch.from_numpy(
         #   t_logprobs).long().view(-1, 1), volatile=not self.training))
        #rewards*=p_actions
        #baseline*=p_actions
        p_actions=t_logprobs[:,0].long()
	advantage=-1*(rewards-baseline)
	batch_size = advantage.size(0)
	seq_length = t_preds.shape[0] / batch_size
	a_index = np.arange(batch_size)
	a_index = a_index.reshape(1, -1).repeat(seq_length, axis=0).flatten()
	advantage = torch.index_select(advantage, 0, torch.from_numpy(a_index))
        #print(advantage.shape)
	#policy_losses = to_gpu(Variable(advantage*p_actions, volatile=p_actions.volatile))*self.rl_weight
        #print(advantage.shape)
	policy_loss=to_gpu(Variable(advantage.long().view(1,-1)))*p_actions
	policy_loss=torch.sum(policy_loss.float())/p_actions.size(0)
	return policy_loss*0.000121392198451


    def output_hook(self, output, sentences, transitions, y_batch=None):
        if not self.training:
            return
        probs = F.softmax(output).data.cpu()
        target = torch.from_numpy(y_batch).long()
        #now we figure out policy loss
        #simple 0-1 loss for rewards
        y = probs.max(1, keepdim=False)[1].cpu()
        rewards = torch.eq(y, torch.Tensor(y_batch).long()).long()
        baseline = torch.zeros(rewards.shape).long()#for now
        advantage = rewards - baseline
        self.policy_loss = self.mc_reinforce(rewards, baseline)


    def forward(
            self,
            sentences,
            transitions,
            y_batch=None,
            use_internal_parser=False,
            validate_transitions=True,
            **kwargs):
        example = self.unwrap(sentences, transitions)

        b, l = example.tokens.size()[:2]

        embeds = self.embed(example.tokens)
        embeds = self.reshape_input(embeds, b, l)
        embeds = self.encode(embeds)
        embeds = self.reshape_context(embeds, b, l)
        self.forward_hook(embeds, b, l)
        embeds = F.dropout(
            embeds,
            self.embedding_dropout_rate,
            training=self.training)

        # Make Buffers
        # _embeds = torch.chunk(to_cpu(embeds), b, 0)
        # _embeds = [torch.chunk(x, l, 0) for x in _embeds]
        # buffers = [list(reversed(x)) for x in _embeds]
        ee = torch.chunk(embeds, b * l, 0)[::-1]##basically broken up into parts!
        bb = []
        for ii in range(b):
            ex = list(ee[ii * l:(ii + 1) * l])
            bb.append(ex)
        buffers = bb[::-1]#Why is x reversed??

        example.bufs = buffers

        h, transition_acc, transition_loss = self.run_spinn(
            example, use_internal_parser, validate_transitions)

        self.spinn_outp = h

        self.transition_acc = transition_acc
        self.transition_loss = transition_loss

        # Build features
        features = self.build_features(h)

        output = self.mlp(features)

        self.output_hook(output, sentences, transitions, y_batch)

        return output
