
class RuleBasedLogicalNetwork:
    def __init__(self):
        self.rules = []  # Stores rules as (conditions, conclusion)
        self.known_facts = set()  # Stores known facts

    def add_rule(self, conditions, conclusion):
        conditions = [tuple(cond) for cond in conditions]  # Convert condition lists to tuples
        conclusion = tuple(conclusion)  # Convert conclusion list to a tuple
        self.rules.append((conditions, conclusion))

    def add_fact(self, fact):
        self.known_facts.add(tuple(fact))  # Convert fact list to tuple before adding

    def infer(self):
        added = True
        while added:
            added = False
            for conditions, conclusion in self.rules:
                if all(cond in self.known_facts for cond in conditions) and conclusion not in self.known_facts:
                    print(f"Rule Applied: {conditions} -> {conclusion}")
                    self.known_facts.add(conclusion)
                    added = True  # Continue inference

    def query_pattern(self, query_pattern):
        query_pattern = tuple(query_pattern)  # Convert to tuple for matching
        def matches(pattern, fact):
            return all(p is None or p == f for p, f in zip(pattern, fact))
        return [fact for fact in self.known_facts if matches(query_pattern, fact)]
