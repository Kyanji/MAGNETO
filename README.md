# Malware Detection Through Images And GAN

### Dataset in dataset/CICDS2017
#### Numero iterate hyperopt:   DeepInsight_train_norn Riga: 58   
 best = fmin(hyperopt_fcn, optimizable_variable, algo=tpe.suggest, max_evals=10, trials=trials)

#### Epoch deep_base_model:229
  model.fit(x_train,
              y_train,
              # validation_data=(x_test, y_test),
              epochs=10,
              batch_size=batch_size,
              use_multiprocessing=False,
              verbose=1)