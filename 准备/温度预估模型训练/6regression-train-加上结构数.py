import pandas as pd
from datetime import datetime
import logging
import joblib
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timezone, timedelta

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


# 0. 读取数据
file_path = r'D:\A_shandong\3项目数据\6数据整理\0-归类整理-dly\温度预估模型\数据\02训练格式整理-机器学习\合并数据-训练格式.csv'  # 替换为实际文件路径
df0 = pd.read_csv(file_path)
save_path = r"D:\A_shandong\temp_model\sample"
os.makedirs(save_path, exist_ok=True)

df= df0.copy()
df['日期时间'] = pd.to_datetime(df['日期时间'])
print(df.columns)

# 1. 数据集拆分
# 选择2023年5月到2024年4月的数据-训练集
train_data = df[
    (df['日期时间'].between(pd.Timestamp('2023-05-01'), pd.Timestamp('2024-04-30'))) &
    (df['结构温度'].between(-15, 60))
].dropna()  # 只删除结构温度缺失的行
train_data = train_data.sample(frac=0.01, random_state=42).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)
# 特征与目标
X_train = train_data[[str(i) for i in range(1, 27)]]
y_train = train_data['结构温度']

# 选择测试集
test_data = df[
    (df['日期时间'].between(pd.Timestamp('2024-05-01'), pd.Timestamp('2025-04-30'))) &
    (df['结构温度'].between(-15, 60))
].dropna()  # 只删除结构温度缺失的行
test_data = test_data.sample(frac=0.001, random_state=42).reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
# 特征与目标
X_test = test_data[[str(i) for i in range(1, 27)]]
y_test = test_data['结构温度']

print(f"X的列名为：{X_train.columns.tolist()}")
print("训练集的shape", X_train.shape, y_train.shape)
print("测试集的shape", X_test.shape, y_test.shape)

# 2. 配置日志记录
logging.basicConfig(
    filename=r'D:\A_shandong\3项目数据\6数据整理\0-归类整理-dly\温度预估模型\加上结构数再做\训练模型\regression_training_log.txt',  # 日志文件名
    level=logging.INFO,                # 记录 INFO 及以上级别的信息
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    encoding='utf-8'
)


# 3. 定义回归模型（添加更多类型）
models = {
    # # K近邻
    # "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance'),
    # # 线性模型
    # "Ridge": Ridge(alpha=1.0),
    # "Lasso": Lasso(alpha=0.001, max_iter=5000),
    # "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
    # # 支持向量机
    # "SVR": SVR(kernel='rbf', C=10, epsilon=0.01, verbose=True),
    # 深度学习
    "MLP": Pipeline([
        ('scaler', StandardScaler()),  # 归一化
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(16, 8),   # 减小网络规模
            alpha=1.0,                   # 增加正则化
            activation='relu',
            solver='adam',
            max_iter=3000,
            learning_rate='adaptive',
            learning_rate_init=0.0005,
            verbose=2,
            early_stopping=True,
            validation_fraction=0.2,     # 增大验证集比例
            n_iter_no_change=150,
            random_state=42
        ))
    ]),
    # 树模型
    # "DecisionTree": DecisionTreeRegressor(
    #     max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42
    # ),
    # "RandomForest": RandomForestRegressor(
    #     n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, verbose=2
    # ),
    # "LightGBM": lgb.LGBMRegressor(
    #     n_estimators=500, learning_rate=0.05, max_depth=10, num_leaves=100, subsample=0.6, verbose=2
    # ),
    # "CatBoost": cb.CatBoostRegressor(
    #     iterations=1500, learning_rate=0.1, depth=10, subsample=0.6, colsample_bylevel=0.9, verbose=2
    # ),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.3, max_depth=10, subsample=0.6, verbosity=2),
}


# 4. 训练和评估模型
results = {}
logging.info(f"start traing ---------------------------------------：")
# logging.info(f'训练观测点分布统计：\n{train_data["面层厚度"].value_counts().to_string()}')
logging.info(f"Train shap: X_train={X_train.shape}, y_train={y_train.shape}")
logging.info(f"Test shape: X_tset={X_test.shape}, y_test={y_test.shape}")



for name, model in models.items():
    print(f"\n🚀 training: {name} ...")
    logging.info(f"trainging {name} ...")
    
    model.fit(X_train, y_train)
    
    # 训练集预测
    y_train_pred = model.predict(X_train)
    # 测试集预测
    y_test_pred = model.predict(X_test)
    
    # 计算训练集和验证集的指标
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    std_train = np.std(y_train - y_train_pred)  # 计算标准差 STD
    mbd_train = np.mean(y_train_pred - y_train)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    std_test = np.std(y_test - y_test_pred)  # 验证集的方差
    mbd_test = np.mean(y_test_pred - y_test)
    
    # 保存模型结果
    results[name] = {
        "Train": {"RMSE": rmse_train, "MAE": mae_train, "R2": r2_train, "STD": std_train, "MBD": mbd_train},
        "Test": {"RMSE": rmse_test, "MAE": mae_test, "R2": r2_test, "STD": std_test, "MBD": mbd_test}
    }
    
    # 打印并记录训练和验证集的结果
    print(f"📊 {name} - 训练集: RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.4f}, STD: {std_train:.4f}, MBD: {mbd_train:.4f}")
    print(f"    {name} - 验证集: RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R²: {r2_test:.4f}, STD: {std_test:.4f}, MBD: {mbd_test:.4f}")
    
    logging.info(f"📊 {name} - train: RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.4f}, STD: {std_train:.4f}, MBD: {mbd_train:.4f}")
    logging.info(f"    {name} - test: RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R²: {r2_test:.4f}, STD: {std_test:.4f}, MBD: {mbd_test:.4f}")
    
    # 保存模型
    # 北京时间 = UTC+8
    beijing_tz = timezone(timedelta(hours=8))
    timestamp = datetime.now(beijing_tz).strftime('%m%d-%H%M')
    # timestamp = datetime.now().strftime('%m%d-%H%M')

    if name == "LightGBM":
        booster = model.booster_  # 获取原始 booster 对象
        model_filename = os.path.join(save_path, f"{name}_model_{timestamp}.txt")
        booster.save_model(model_filename)

    elif name == "XGBoost":
        booster = model.get_booster()
        model_filename = os.path.join(save_path, f"{name}_model_{timestamp}.json")
        booster.save_model(model_filename)

    elif name == "CatBoost":
        model_filename = os.path.join(save_path, f"{name}_model_{timestamp}.cbm")
        model.save_model(model_filename)

    # 在您的训练代码中，修改保存部分：
    elif name == "MLP":
        # 保存整个 Pipeline，而不是单独的 MLP
        model_filename = os.path.join(save_path, f"{name}_model_{timestamp}.pkl")
        joblib.dump(model, model_filename)  # 保存整个 Pipeline

    elif name in ["DecisionTree", "RandomForest", "KNN", "Ridge", "Lasso", "ElasticNet", "SVR", "HistGB"]:
        model_filename = os.path.join(save_path, f"{name}_model_{timestamp}.pkl")
        joblib.dump(model, model_filename)
        
    else:  # RandomForest
        model_filename = os.path.join(save_path, f"{name}_model_{timestamp}.pkl")
        joblib.dump(model, model_filename)

    print(f"💾 {name} 模型已保存为 {model_filename}")
    logging.info(f"{name} model saved to {model_filename}")

