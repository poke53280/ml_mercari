
import numpy as np
import pandas as pd
import dowhy.api


N = 5000

z = np.random.uniform(size = N)
d = np.random.binomial(1, p = 1/ (1 + np.exp(-5 * z)))
y = 2 * z + d + 0.1 * np.random.normal(size = N)
df = pd.DataFrame({'Z': z, 'D': d, 'Y': y})


df[df.D == 1].Y.mean() - df[df.D == 0].Y.mean()

causes = ['D']
outcomes  = ['Y']
common_causes = ['Z']

model = CausalModel(df, causes, outcomes, common_causes=common_causes)

identification = model.identify_effect()

from dowhy.do_samplers.weighting_sampler import WeightingSampler

sampler = WeightingSampler(df, causal_model=model, keep_original_treatment=True, variable_types={'D': 'b', 'Z': 'c', 'Y': 'c'})


sampler

interventional_df = sampler.do_sample(None)


(interventional_df[interventional_df.D == 1].mean() - interventional_df[interventional_df.D == 0].mean())['Y']