
#

# SEE: https://causalinference.gitlab.io/kdd-tutorial/

import numpy as np
import pandas as pd
import dowhy
import networkx


from dowhy.do_why import CausalModel
import dowhy.datasets


data = dowhy.datasets.linear_dataset(beta=10, num_common_causes =5, num_instruments = 2, num_samples=10000, treatment_is_binary=True)

df = data["df"]

g = data["gml_graph"]

n = networkx.parse_gml(g)

networkx.write_gexf(n, "C:\\Users\\T149900\\pim.gexf", encoding='utf-8', prettyprint=True, version='1.1draft')

# Create a causal model from the data and given graph.
model = CausalModel(data=df, treatment=data["treatment_name"], outcome=data["outcome_name"], graph=data["gml_graph"])


# Identify causal effect and return target estimands
identified_estimand = model.identify_effect()


print(identified_estimand)


causal_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")

print(causal_estimate)
print("Causal Estimate is " + str(causal_estimate.value))