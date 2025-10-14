##################### Parameters to vary ############################
## Tail heaviness: tails of error distribution F (Student-t degrees of freedom) dfs
## Aspect ratio: \gamma = p/n controls the dimensionality regime \gamma s
## Design structure: correlation among predictors \rho s
## Signal-to-noise ratio: overall signal strength snrs
#####################################################################

PARAM_GRID = {
    "dfs": [1, 2, 3, 20, float("inf")],
    "gammas": [0.2, 0.5, 0.8],   # p/n
    "rhos": [0.0, 0.25, 0.5, 0.75, 1.0],
    "snrs": [1, 5, 10],
    "reps": 30,
    "n": 500,                    # base n
}