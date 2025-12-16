import numpy as np
import torch
from sksurv.metrics import concordance_index_ipcw, brier_score
from sksurv.util import Surv

class SurvITE_Evaluator:
    def __init__(self, model, train_data, test_data):
        """
        Args:
            model: The trained SurvITE PyTorch model.
            train_data: Tuple (x, y, t, a) for training set (needed for IPCW censoring distribution).
            test_data: Tuple (x, y, t, a) for testing set.
        """
        self.model = model
        self.x_tr, self.y_tr, self.t_tr, self.a_tr = train_data
        self.x_te, self.y_te, self.t_te, self.a_te = test_data
        
        # Format data for scikit-survival: (Status, Time) structured array
        self.struc_tr = Surv.from_arrays(event=self.y_tr.flatten().astype(bool), time=self.t_tr.flatten())
        self.struc_te = Surv.from_arrays(event=self.y_te.flatten().astype(bool), time=self.t_te.flatten())
        
        # Get Factual Predictions for Test Set
        self.surv_probs = self._get_factual_survival(self.x_te, self.a_te)
        
        # Define evaluation time grid (based on model's T_max)
        # We only evaluate within the range of observed test times to avoid extrapolation errors
        self.t_min = self.t_te.min()
        self.t_max = self.t_te.max()
        self.time_grid = np.arange(int(self.t_min), int(self.t_max))


        if len(self.time_grid) == 0:
            print("Warning: No valid evaluation time points found.")

    def _get_factual_survival(self, x, a):
        """
        Extracts S(t | A_factual) from the model.
        """
        # Get predictions for both arms
        surv1 = self.model.predict_survival_A1(x)
        surv0 = self.model.predict_survival_A0(x)
        
        # Select the curve corresponding to the actual treatment 'a'
        surv_factual = np.zeros_like(surv1)
        
        # Identify indices
        idx1 = (a.flatten() == 1)
        idx0 = (a.flatten() == 0)
        
        surv_factual[idx1] = surv1[idx1]
        surv_factual[idx0] = surv0[idx0]
        
        return surv_factual

    def calculate_c_index(self, eval_time=None):
        """
        Calculates Harrell's C-index (IPCW) at a specific time horizon.
        If eval_time is None, it uses the median follow-up time.
        """
        if eval_time is None:
            eval_time = np.median(self.t_te)
            
        # Ensure eval_time is within model limits
        eval_idx = int(eval_time)
        if eval_idx >= self.surv_probs.shape[1]:
            eval_idx = self.surv_probs.shape[1] - 1

        # Prediction for C-index is Risk. Risk = 1 - Survival
        # Higher risk score should correspond to shorter survival time.
        risk_scores = 1.0 - self.surv_probs[:, eval_idx]

        c_index, concordant, discordant, tied_risk, tied_time = concordance_index_ipcw(
            self.struc_tr, 
            self.struc_te, 
            risk_scores, 
            tau=eval_time 
        )
        return c_index

    def calculate_ibs(self):
        """
        Calculates the Integrated Brier Score (IBS).
        """
        brier_scores = []
        valid_times = []
        
        for t in self.time_grid:
            # Check if time index exists in model output columns
            if t >= self.surv_probs.shape[1]:
                continue
                
            # Get survival probability at time t for all test patients
            preds_at_t = self.surv_probs[:, int(t)]
            
            # Calculate Brier Score at this specific time point
            # IPCW weighting requires training set distribution
            score = brier_score(
                self.struc_tr, 
                self.struc_te, 
                preds_at_t, 
                times=t
            )[1][0] # returns (times, scores), we take score
            
            brier_scores.append(score)
            valid_times.append(t)
            
        # Integrate using Trapezoidal rule
        # IBS = Integral(BS(t) dt) / (t_max - t_min)
        valid_times = np.array(valid_times)
        brier_scores = np.array(brier_scores)
        
        integral = np.trapz(brier_scores, valid_times)
        ibs = integral / (valid_times.max() - valid_times.min())
        
        return ibs, valid_times, brier_scores
