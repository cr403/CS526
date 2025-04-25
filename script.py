import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from collections import defaultdict
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import plot_tree
from skorch.callbacks import EarlyStopping
from pdb import set_trace as bb

# === Data Preprocessing ===
df = pd.read_csv("./datasets/FastFoodNutritionMenuV3.csv")
df.columns = [col.replace('\n', ' ').strip() for col in df.columns]

selected_features = [
    'Calories', 'Trans Fat (g)', 'Cholesterol (mg)', 'Sodium  (mg)',
    'Carbs (g)', 'Fiber (g)', 'Sugars (g)', 'Protein (g)'
]
cols_to_convert = selected_features + ['Weight Watchers Pnts']
df[cols_to_convert] = df[cols_to_convert].replace('[^\d.]', '', regex=True).replace('', np.nan).astype(float)
df = df.dropna(subset=['Weight Watchers Pnts'])
df.fillna(df.mean(numeric_only=True), inplace=True)

X = df[selected_features]
y = df['Weight Watchers Pnts']

# === OLS Analysis ===
print("\n==== 📊 Statsmodels OLS Regression Analysis ====")
X_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_const).fit()
print(ols_model.summary())

# === PyTorch Neural Network Model ===
class ReshapeMSELoss(nn.MSELoss):
    def forward(self, input, target):
        if target.dim() == 1:
            target = target.view(-1, 1)
        return super().forward(input, target)

class DropoutNet(nn.Module):
    def __init__(self, input_dim, n_hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(n_hidden, n_hidden // 2)
        self.drop2 = nn.Dropout(0.1)
        self.out = nn.Linear(n_hidden // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.out(x)
        return x

nn_model = NeuralNetRegressor(
    DropoutNet,
    module__input_dim=X.shape[1],
    criterion=ReshapeMSELoss,
    optimizer=torch.optim.Adam,
    max_epochs=100,
    lr=0.001,
    batch_size=64,
    device='cpu',
)

# === Models to Compare ===
lasso_cv = LassoCV(alphas=np.logspace(-3, 2, 100), cv=5)
ridge_cv = RidgeCV(alphas=np.logspace(-3, 2, 100), cv=5)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": ridge_cv,
    "Lasso": lasso_cv,
    "SVR": SVR(kernel='linear'),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, verbosity=0),
    "PyTorch NeuralNet": nn_model
}

# === Training and Evaluation ===
results = defaultdict(list)
train_mse_record = defaultdict(list)
test_mse_record = defaultdict(list)
residuals_dict = defaultdict(list)

X_np = X.values.astype(np.float32)
y_np = y.values.astype(np.float32).reshape(-1, 1)

for run in range(1):
    print(f"\n==== Training Run {run + 1} ====")
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42 + run
    )
    for name, model in models.items():
        if name != "PyTorch NeuralNet":
            model.fit(X_train, y_train.ravel())
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
                        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        train_mse_record[name].append(train_mse)
        test_mse_record[name].append(test_mse)
        results[f"{name}_MSE"].append(test_mse)
        results[f"{name}_MAE"].append(mae)
        results[f"{name}_R2"].append(r2)
        residuals = y_test.ravel() - y_pred_test
        residuals_dict[name].append(residuals)
        print(f"📊 {name}: MSE={test_mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        if name == "Linear Regression":
            n_params = model.coef_.size + 1
        elif name == "Ridge":
            n_params = model.coef_.size + 1
        elif name == "Lasso":
            n_params = model.coef_.size + 1
        elif name == "SVR":
            n_params = len(model.support_) * X.shape[1] + 1
            print(f"📌 SVR Number of Support Vectors: {len(model.support_)}")
        elif name == "Random Forest":
            n_params = sum(tree.tree_.node_count for tree in model.estimators_)
        elif name == "XGBoost":
            booster_df = model.get_booster().trees_to_dataframe()
            n_params = booster_df.shape[0]
        elif name == "PyTorch NeuralNet":
            n_params = sum(p.numel() for p in model.module_.parameters() if p.requires_grad)
        else:
            n_params = np.nan

        print(f"🧮 {name} Estimated Number of Parameters: {n_params}")

# === Lasso: Alpha Grid Search + Regression Equation ===
lasso_model = models["Lasso"]
alphas = lasso_model.alphas_
mae_list = []
for alpha in alphas:
    model = Lasso(alpha=alpha)
    preds = cross_val_predict(model, X, y, cv=5)
    mae_list.append(mean_absolute_error(y, preds))

best_index = np.argmin(mae_list)
best_alpha_lasso = alphas[best_index]
best_mae_lasso = mae_list[best_index]
final_lasso = Lasso(alpha=best_alpha_lasso).fit(X, y)
coef_lasso = final_lasso.coef_

print(f"\n🔍 Best alpha for Lasso: {best_alpha_lasso}")
print("📘 Lasso Regression Equation:")
intercept = final_lasso.intercept_
terms = [f"{coef_lasso[i]:.3f}*{selected_features[i]}" for i in range(len(coef_lasso)) if abs(coef_lasso[i]) > 1e-6]
formula = f"y = {intercept:.3f} + " + " + ".join(terms)
print(formula)

plt.figure(figsize=(8, 4))
plt.plot(alphas, mae_list, label="5-fold CV MAE")
plt.scatter([best_alpha_lasso], [best_mae_lasso], color='red', label='Best MAE')
plt.text(best_alpha_lasso, best_mae_lasso, f'({best_alpha_lasso:.4f}, {best_mae_lasso:.4f})', color='red')
plt.xscale('log')
plt.xlabel(r"$\alpha$")
plt.ylabel("5-fold CV Mean MAE")
plt.title(r"Lasso: MAE-Based Grid Search over $\alpha$")
plt.legend()
plt.tight_layout()
plt.savefig("lasso_alpha_mae.png")
print("✅ Lasso MAE curve saved as 'lasso_alpha_mae.png'")

# === Ridge: Alpha Grid Search + Regression Equation ===
ridge_model = models["Ridge"]
alphas = ridge_model.alphas
mae_list = []
for alpha in alphas:
    model = Ridge(alpha=alpha)
    preds = cross_val_predict(model, X, y, cv=5)
    mae_list.append(mean_absolute_error(y, preds))

best_index = np.argmin(mae_list)
best_alpha_ridge = alphas[best_index]
best_mae_ridge = mae_list[best_index]
final_ridge = Ridge(alpha=best_alpha_ridge).fit(X, y)
coef_ridge = final_ridge.coef_

print(f"\n🔍 Best alpha for Ridge: {best_alpha_ridge}")
print("📘 Ridge Regression Equation:")
intercept = final_ridge.intercept_
terms = [f"{coef_ridge[i]:.3f}*{selected_features[i]}" for i in range(len(coef_ridge)) if abs(coef_ridge[i]) > 1e-6]
formula = f"y = {intercept:.3f} + " + " + ".join(terms)
print(formula)

plt.figure(figsize=(8, 4))
plt.plot(alphas, mae_list, label="5-fold CV MAE")
plt.scatter([best_alpha_ridge], [best_mae_ridge], color='red', label='Best MAE')
plt.text(best_alpha_ridge, best_mae_ridge, f'({best_alpha_ridge:.4f}, {best_mae_ridge:.4f})', color='red')
plt.xscale('log')
plt.xlabel(r"$\alpha$")
plt.ylabel("5-fold CV Mean MAE")
plt.title(r"Ridge: MAE-Based Grid Search over $\alpha$")
plt.legend()
plt.tight_layout()
plt.savefig("ridge_alpha_mae.png")
print("✅ Ridge MAE curve saved as 'ridge_alpha_mae.png'")

# === SVR (Linear Kernel) 3D Grid Search ===
svr_pipeline = Pipeline([("svr", SVR(kernel="linear"))])
param_grid = {
    "svr__C": np.logspace(-2, 2, 10),
    "svr__epsilon": np.logspace(-2, 0, 10)
}
grid_search = GridSearchCV(
    estimator=svr_pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X, y)

best_C = grid_search.best_params_["svr__C"]
best_epsilon = grid_search.best_params_["svr__epsilon"]
best_score = -grid_search.best_score_
print(f"\n🔧 Best parameters for SVR: C={best_C}, epsilon={best_epsilon}, MAE={best_score:.4f}")

svr_grid_results = pd.DataFrame(grid_search.cv_results_)
svr_grid_results["C"] = svr_grid_results["param_svr__C"].astype(float)
svr_grid_results["epsilon"] = svr_grid_results["param_svr__epsilon"].astype(float)
svr_grid_results["MAE"] = -svr_grid_results["mean_test_score"]

C_vals = sorted(svr_grid_results["C"].unique())
eps_vals = sorted(svr_grid_results["epsilon"].unique())
C_grid, eps_grid = np.meshgrid(C_vals, eps_vals)
mae_grid = np.empty_like(C_grid)
for i in range(len(eps_vals)):
    for j in range(len(C_vals)):
        mae = svr_grid_results[
            (svr_grid_results["C"] == C_vals[j]) & (svr_grid_results["epsilon"] == eps_vals[i])
        ]["MAE"].values[0]
        mae_grid[i, j] = mae

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(np.log10(C_grid), np.log10(eps_grid), mae_grid,
                       cmap='viridis', edgecolor='k', alpha=0.8)
ax.scatter(np.log10(best_C), np.log10(best_epsilon), best_score,
           color='red', s=60, label='Best MAE')
ax.text(np.log10(best_C), np.log10(best_epsilon), best_score,
        f"({best_C:.4f}, {best_epsilon:.4f}, {best_score:.4f})", color='red')
ax.set_xlabel(r"$\log_{10}(C)$")
ax.set_ylabel(r"$\log_{10}(\epsilon)$")
ax.set_zlabel("5-fold CV Mean MAE")
ax.set_title(r"SVR (Linear Kernel): MAE-Based Grid Search over $C$ and $\epsilon$")
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.legend()
plt.tight_layout()
plt.savefig("svr_gridsearch_mae_3d.png")
print("✅ SVR 3D grid search plot saved as 'svr_gridsearch_mae_3d.png'")

# === Visualize Representative Tree from Random Forest ===
rf_model = models["Random Forest"]
rf_model.fit(X, y)
forest_importance = rf_model.feature_importances_
tree_importances = [tree.feature_importances_ for tree in rf_model.estimators_]
best_tree_idx = np.argmin([np.linalg.norm(imp - forest_importance) for imp in tree_importances])
plt.figure(figsize=(16, 6))
plot_tree(rf_model.estimators_[best_tree_idx],
          feature_names=selected_features,
          filled=True, rounded=True, fontsize=10, max_depth=3)
plt.title(f"Most Representative Tree from Random Forest (Tree #{best_tree_idx})")
plt.tight_layout()
plt.savefig("rf_representative_tree.png")
print(f"✅ Random Forest representative tree saved as 'rf_representative_tree.png'")

# === Average Model Performance Metrics (1 Run Example) ===
print("\n==== 📈 Average Model Performance ====")
for name in models:
    for metric in ['MSE', 'MAE', 'R2']:
        scores = results[f"{name}_{metric}"]
        print(f"{name} - {metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# === Test Residuals Boxplot for All Models ===
import seaborn as sns

model_names = list(models.keys())
all_residuals = []
for name in model_names:
    residuals = np.concatenate(residuals_dict[name])
    all_residuals.append(residuals)

residuals_df = pd.DataFrame({
    "model": np.repeat(model_names, [len(r) for r in all_residuals]),
    "residual": np.concatenate(all_residuals)
})

plt.figure(figsize=(12, 6))
ax = sns.boxplot(x="model", y="residual", data=residuals_df, showmeans=True, showfliers=False, palette="Set2")
plt.title("Test Residuals (y - y_pred) Across Models")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("model_residuals_boxplot.png")
print("✅ Residuals boxplot saved as 'model_residuals_boxplot.png'")
