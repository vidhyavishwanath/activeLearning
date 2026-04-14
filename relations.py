"""
relations.py
------------
Sits between shape_detection.py and the robot behaviours.

  shape_detection.detect_shapes(session)
        |
        v
  get_relation(shapes)          -- top / bottom from vertical position
        |
        v
  ConceptLearner.update()       -- supervised: record label
  ConceptLearner.confidence()   -- how sure is the robot?
  ConceptLearner.query()        -- active: what piece to ask to swap?
"""


# ── Relation extraction ────────────────────────────────────────────────────────

def get_relation(shapes):
    """
    Given detect_shapes() output, return the compound object as a dict.

    Smaller cy_norm = higher in the camera frame = the top piece.
    Returns None if fewer than 2 shapes are detected.

    Example
    -------
    shapes = [
        {'color': 'red',  'cy_norm': 0.30, ...},
        {'color': 'blue', 'cy_norm': 0.65, ...},
    ]
    get_relation(shapes) -> {'top': 'red', 'bottom': 'blue'}
    """
    if len(shapes) < 2:
        return None

    sorted_shapes = sorted(shapes, key=lambda s: s['cy_norm'])
    return {
        'top':    sorted_shapes[0]['color'],
        'bottom': sorted_shapes[-1]['color'],
    }


def relation_to_str(relation):
    """Human-readable: 'red on blue'."""
    if relation is None:
        return 'nothing detected'
    return '{} on {}'.format(relation['top'], relation['bottom'])


# ── Concept learner ────────────────────────────────────────────────────────────

class ConceptLearner(object):
    """
    Online learner for a single concept (e.g. 'house').
    Works in supervised or active learning mode.

    Supervised mode (HOUSE)
    -----------------------
    - Participant labels each config as positive or negative.
    - Robot records the label and updates its confidence.
    - Robot never asks questions on its own.

    Active learning mode (SNOWMAN)
    ------------------------------
    - Same as supervised PLUS:
    - After every participant statement the robot generates a query
      (asks the participant to swap one piece) to get the most
      informative next example.

    Usage
    -----
    learner = ConceptLearner(mode='supervised')

    relation = get_relation(detect_shapes(session))
    learner.update(relation, is_positive=True)   # participant said "this is a house"
    print(learner.confidence(relation))          # 0.0 – 1.0
    print(learner.step)                          # how many examples seen

    # active mode only:
    learner = ConceptLearner(mode='active')
    learner.update(relation, is_positive=True)
    query = learner.generate_query(relation)
    # -> {'position': 'bottom', 'swap_to': 'green'}
    # robot says: "Can you replace the bottom piece with a green one?"
    """

    POSITIONS = ('top', 'bottom')

    def __init__(self, mode='supervised'):
        if mode not in ('supervised', 'active'):
            raise ValueError("mode must be 'supervised' or 'active'")
        self.mode     = mode
        self.step     = 0          # total examples seen
        self.n_queries = 0         # how many queries the robot has made

        # Counts: how many times each (position, color) pair appeared in
        # positive vs negative examples.
        # Structure: { 'top':    {'red': [pos_count, neg_count], ...},
        #              'bottom': {'blue': [pos_count, neg_count], ...} }
        self._counts = {pos: {} for pos in self.POSITIONS}

    # ── Learning ──────────────────────────────────────────────────────────────

    def update(self, relation, is_positive):
        """
        Record a labeled example.

        Parameters
        ----------
        relation    : dict from get_relation() -- {'top': color, 'bottom': color}
        is_positive : bool -- True if participant said 'this IS a [concept]'
        """
        if relation is None:
            return

        self.step += 1
        inc = 0 if is_positive else 1   # index: 0=positive count, 1=negative count

        for pos in self.POSITIONS:
            color = relation[pos]
            if color not in self._counts[pos]:
                self._counts[pos][color] = [0, 0]
            self._counts[pos][color][0 if is_positive else 1] += 1

    # ── Confidence ────────────────────────────────────────────────────────────

    def confidence(self, relation):
        """
        Estimate how confident the robot is that the current relation matches
        the concept it is learning. Returns a float in [0, 1].

        For each position (top, bottom):
          - Look up how often that color appeared in positive vs all examples.
          - Average across both positions.

        If a color has never been seen, that position contributes 0.5
        (maximum uncertainty).
        """
        if relation is None or self.step == 0:
            return 0.5

        scores = []
        for pos in self.POSITIONS:
            color = relation[pos]
            counts = self._counts[pos].get(color, None)
            if counts is None:
                scores.append(0.5)          # never seen this color here
            else:
                pos_count, neg_count = counts
                total = pos_count + neg_count
                scores.append(pos_count / float(total))

        return sum(scores) / len(scores)

    # ── Active learning: query generation ─────────────────────────────────────

    def should_query(self):
        """
        SNOWMAN mode: robot queries after every participant statement.
        ALIEN mode could check confidence here instead.
        """
        return self.mode == 'active'

    def generate_query(self, current_relation):
        """
        Decide which piece (top or bottom) to ask the participant to swap,
        and suggest a color that hasn't been seen in that position yet.

        Strategy: pick the position the robot is LEAST certain about,
        then suggest a color it has never seen there (maximally informative).

        Returns
        -------
        dict: {'position': 'top' | 'bottom', 'swap_to': color_name | None}
        or None if no useful query can be generated.

        The caller uses this to build the sentence:
          "Can you replace the [position] piece with a [swap_to] one?"
        or if swap_to is None:
          "Can you replace the [position] piece with something different?"
        """
        if current_relation is None:
            return None

        # Score each position by uncertainty (closer to 0.5 = more uncertain)
        position_uncertainty = {}
        for pos in self.POSITIONS:
            color = current_relation[pos]
            counts = self._counts[pos].get(color, None)
            if counts is None:
                uncertainty = 0.5
            else:
                pos_count, neg_count = counts
                total = pos_count + neg_count
                p = pos_count / float(total)
                uncertainty = 1.0 - abs(p - 0.5) * 2   # 1.0 = maximally uncertain

            position_uncertainty[pos] = uncertainty

        # Pick the most uncertain position to ask about
        query_position = max(position_uncertainty, key=position_uncertainty.get)

        # Suggest a color not yet seen in that position (novel = informative)
        seen_colors = set(self._counts[query_position].keys())
        all_colors  = {'red', 'blue', 'green', 'yellow'}
        unseen      = all_colors - seen_colors
        swap_to     = next(iter(unseen)) if unseen else None

        self.n_queries += 1
        return {
            'position': query_position,
            'swap_to':  swap_to,
        }

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self):
        """Print what the robot has learned so far."""
        print("Steps: {}  Queries: {}".format(self.step, self.n_queries))
        for pos in self.POSITIONS:
            for color, (p, n) in self._counts[pos].items():
                total = p + n
                print("  {} {}:  {} pos  {} neg  ({:.0%} positive)".format(
                    pos, color, p, n, p / float(total)))
