# This code has been adapted from PFLlib https://github.com/TsingZ0/PFLlib
import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        # save results after local training (FedAvg-FT)
        self.rs_test_acc_after = []
        self.rs_train_acc_after = []
        self.rs_val_acc_after = []

        # save results before local training (FedAvg)
        self.rs_test_acc_before = []
        self.rs_train_acc_before = []
        self.rs_val_acc_before = []

        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(self.rs_test_acc_before, self.rs_train_acc_before, val=self.rs_val_acc_before)

            for client in self.selected_clients:
                client.train()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized model")
                self.evaluate(self.rs_test_acc_after, self.rs_train_acc_after, val=self.rs_val_acc_after)

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # result of FedAvg
        """
        print("\nBest accuracy for before local train.")
        print(max(self.rs_test_acc_before))
        print("\naccuracy before local train.")
        print(self.rs_test_acc_before)
        """
        print("\nBest val accuracy for before local train.")
        print(max(self.rs_val_acc_before))
        print("\nval accuracy before local train.")
        print(self.rs_val_acc_before)
        best_val_index_before = np.argmax(np.array(self.rs_val_acc_before))

        # result of FedAvg-FT
        """
        print("\nBest accuracy for after local train.")
        print(max(self.rs_test_acc_after))
        print("\naccuracy after local train.")
        print(self.rs_test_acc_after)
        """
        print("\nBest val accuracy for after local train.")
        print(max(self.rs_val_acc_after))
        print("\nval accuracy after local train.")
        print(self.rs_val_acc_after)
        best_val_index = np.argmax(np.array(self.rs_val_acc_after))

        # result of FedAvg
        #print("FedAvg_best test accuracy for after local train", max(self.rs_test_acc_before))
        print("FedAvg_best_val_rounds_index", best_val_index_before)
        print("FedAvg_best_val_acc", self.rs_val_acc_before[best_val_index_before])
        print("FedAvg_best_val_model_test_acc", self.rs_test_acc_before[best_val_index_before])

        # result of FedAvg-FT
        #print("FedAvg-FT_best test accuracy for after local train", max(self.rs_test_acc_after))
        print("FedAvg-FT_best_val_rounds_index", best_val_index)
        print("FedAvg-FT_best_val_acc", self.rs_val_acc_after[best_val_index])
        print("FedAvg-FT_best_val_model_test_acc", self.rs_test_acc_after[best_val_index])

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
