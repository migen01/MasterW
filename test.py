from lnn import And, Fact, Proposition,Model

# Rules
EPL = Proposition("Santa Monica, California")
BornInEngland = Proposition("California")
AND = And(EPL, BornInEngland)
model = Model()

# Data
EPL.add_data(Fact.TRUE)
BornInEngland.add_data(Fact.UNKNOWN)
model.add_data([AND])

# Reasoning
AND.upward()
AND.print()

model.infer()

# Check inference results
print("WetGround truth value:", model[BornInEngland].state)