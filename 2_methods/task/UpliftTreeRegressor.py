class UpliftTreeRegressor:
    
    def __init__(
        self,
        max_depth: int = 3, # максимальная глубина дерева.
        min_samples_leaf: int = 1000, # минимальное необходимое число обучающих объектов в листе дерева.
        min_samples_leaf_treated: int = 300, # минимальное необходимое число обучающих объектов с T=1 в листе дерева.
        min_samples_leaf_control: int = 300, # минимальное необходимое число обучающих объектов с T=0 в листе дерева.
    ):
        
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        
    def calc_ddp(self, y, treatment):
        import numpy as np
        return np.sum(y*treatment)/np.sum(treatment) - np.sum(y * (1 - treatment))/np.sum(1 - treatment)
        
    def _best_split(self, X, y, treatment):
        """Find the best split for a node.
        Returns:
            best_idx: Index of the feature for best split, or None if no split is found.
            best_thr: Threshold to use for the split, or None if no split is found.
        """
        import numpy as np

        # DDP of current node.
        best_ddp = self.calc_ddp(y, treatment)
        
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            
            column_values = X[:, idx]
            
            unique_values = np.unique(column_values)
            if len(unique_values) > 10:
                percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
            else:
                percentiles = np.percentile(unique_values, [10, 50, 90])
            
            thresholds = np.unique(percentiles)

            for thr in thresholds: 
                
                left_idx = column_values <= thr
 
                treatment_left = treatment[left_idx]
                treatment_right = treatment[~left_idx]
                
                y_left = y[left_idx]
                y_right = y[~left_idx]
                
                len_y_left = len(y_left)
                len_y_right = len(y_right)
                                
                samples_leaf_treated_left = np.sum(treatment_left == 1)
                samples_leaf_control_left = np.sum(treatment_left == 0)
                
                samples_leaf_treated_right = np.sum(treatment_right == 1)
                samples_leaf_control_right = np.sum(treatment_right == 0)
                
                if  not ((samples_leaf_treated_left >= self.min_samples_leaf_treated) and \
                (samples_leaf_treated_right >= self.min_samples_leaf_treated) and \
                (samples_leaf_control_left >= self.min_samples_leaf_control) and \
                (samples_leaf_control_right >= self.min_samples_leaf_control) and \
                (len_y_left >= self.min_samples_leaf) and \
                (len_y_right >= self.min_samples_leaf)):
                    continue

                ddp_left = self.calc_ddp(y_left, treatment_left)
                ddp_right = self.calc_ddp(y_right, treatment_right)
            
                ddp = np.abs(ddp_left - ddp_right)
                    
                if ddp > best_ddp:
#                     print(f'{idx}...{thr}...{ddp}')
                    best_ddp = ddp
                    best_idx = idx
                    best_thr = thr
            
        return best_idx, best_thr
    
    def _grow_tree(self, X, y, treatment, depth=0):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node. The predicted class is the one with
        # largest population.
        
        import numpy as np

        ATE = np.nanmean(y[treatment == 1]) - np.nanmean(y[treatment == 0])
        
        node = self.Node(
            n_items=len(y),
            ATE=ATE,
        )

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
        
            idx, thr = self._best_split(X, y, treatment)
            
            if idx is not None:
                
                indices_left = X[:, idx] <= thr
                
                X_left, y_left, treatment_left = X[indices_left], y[indices_left], treatment[indices_left]
                X_right, y_right, treatment_right = X[~indices_left], y[~indices_left], treatment[~indices_left]
                    
#                 node.split_feat = f'feat{idx}'
                node.split_feat = idx
                node.split_threshold = thr
                node.left = self._grow_tree(X_left, y_left, treatment_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, treatment_right, depth + 1)
            else:
                return node
                    
        return node
    
    def fit(
        self,
        X,#: self.np.ndarray, # массив (n * k) с признаками.
        y,#: self.np.ndarray, # массив (n) с целевой переменной.
        treatment,#: self.np.ndarray, # массив (n) с флагом воздействия.
    ): # -> None:
                
        # fit the model
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y, treatment)
    
    def predict(self, 
                X,#: self.np.ndarray
    ):# -> self.Iterable[float]:
        # compute predictions
        
        import numpy as np
        
        return np.array([self._predict(inputs) for inputs in X])
    
    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.split_feat] <= node.split_threshold:
#             split_feat = int(node.split_feat.replace('feat', ''))
#             if inputs[split_feat] <= node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node.ATE
    
    class Node:
        def __init__(self, 
                     n_items, 
    #                  num_samples_per_class, 
                     ATE
                    ): 
            self.n_items = n_items
    #         self.num_samples_per_class = num_samples_per_class
            self.ATE = ATE
            self.split_feat = None
            self.split_threshold = None
            self.left = None
            self.right = None