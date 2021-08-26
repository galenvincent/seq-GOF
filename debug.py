import sequentialGOF as gof

ns = gof.Simulation(gof.NormalSequence(0, 1), gof.NormalSequence(2, 1), 15, 20, 25, 25)

knn = gof.KnnRegressor(variables=['x'])

ns.test(knn, progress_bar=True)