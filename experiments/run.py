from experiments import lr, lr_multi, arima, arima_multi, light_gbm, light_gbm_multi, rf, rf_multi, \
    n_beats, n_beats_multi, lstm, lstm_multi, tcn, tcn_multi, transformer, transformer_multi, prophet_, prophet_multi

if __name__ == '__main__':
    arima.run_tests()
    arima_multi.run_tests()

    light_gbm.run_tests()
    light_gbm_multi.run_tests()

    prophet_.run_tests()
    prophet_multi.run_tests()

    lr.run_tests()
    lr_multi.run_tests()

    rf.run_tests()
    rf_multi.run_tests()

    n_beats.run_tests()
    n_beats_multi.run_tests()

    lstm.run_tests()
    lstm_multi.run_tests()

    tcn.run_tests()
    tcn_multi.run_tests()

    transformer.run_tests()
    transformer_multi.run_tests()
