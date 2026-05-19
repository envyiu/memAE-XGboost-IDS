Cấu trúc Kiến trúc và Triển khai Mã nguồn Tích hợp TabTransformer và XGBoost cho Hệ thống Phát hiện Bất thường Luồng Mạng (IDS)Triển khai một Hệ thống Phát hiện Xâm nhập (IDS) có khả năng nhận diện các cuộc tấn công zero-day trên dữ liệu dạng bảng đòi hỏi một kiến trúc kết hợp khả năng nhúng ngữ cảnh của mô hình học sâu và sức mạnh phân chia ranh giới quyết định của thuật toán cây tăng cường gradient. Thay vì triển khai TabTransformer như một mạng nơ-ron từ đầu đến cuối (end-to-end) với lớp Multi-Layer Perceptron (MLP) ở tầng phân loại cuối cùng, hệ thống này được thiết kế theo dạng đường ống (pipeline) hai giai đoạn. Giai đoạn đầu tiên sử dụng TabTransformer để biến đổi các đặc trưng phân loại thô và đặc trưng liên tục thành một không gian biểu diễn tiềm ẩn (latent representation space) dày đặc và giàu ngữ cảnh. Giai đoạn thứ hai loại bỏ lớp MLP truyền thống, thay vào đó truyền trực tiếp các ma trận biểu diễn này vào mô hình eXtreme Gradient Boosting (XGBoost) để thực hiện quá trình phân loại nhị phân hoặc đa lớp. Sự kết hợp này giải quyết dứt điểm điểm yếu của các mạng nơ-ron thuần túy trong việc xử lý ranh giới quyết định cứng trên dữ liệu phi cấu trúc, đồng thời vượt qua giới hạn của XGBoost trong việc nắm bắt tương quan chéo giữa các đặc trưng phân loại có tính cardinality cao.Triển khai Mã nguồn TabTransformer làm Bộ trích xuất Đặc trưng (Feature Extractor)Kiến trúc mã nguồn của TabTransformer được tùy chỉnh trực tiếp từ nền tảng PyTorch, dựa trên cấu trúc cốt lõi của thư viện tab-transformer-pytorch do cộng đồng mã nguồn mở (lucidrains) phát triển. 1  Trong hệ thống IDS, mã nguồn này không được sử dụng để xuất ra logits dự đoán, mà được tinh chỉnh lại phương thức forward để trả về mảng tensor chứa các đặc trưng đã qua nhúng (embeddings) và chuẩn hóa.Đoạn mã dưới đây trình bày toàn bộ việc xây dựng khối Transformer, cơ chế tự chú ý (Self-Attention), và lớp TabTransformer tùy chỉnh. Việc quản lý các tensor trong mô hình này tuân thủ nghiêm ngặt các quy chuẩn về kích thước lô (batch size) và số chiều của bộ nhớ.Pythonimport torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        sim = (q @ k.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class TransformerBlock(nn.Module):
    def __init__(self, depth, dim, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList())

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TabTransformerFeatureExtractor(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim=32,
        depth=6,
        heads=8,
        dim_head=16,
        attn_dropout=0.1,
        ff_dropout=0.1,
        continuous_mean_std=None
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'Số lượng giá trị duy nhất của mỗi đặc trưng phân loại phải lớn hơn 0'
        
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_continuous = num_continuous
        
        self.embeds = nn.Embedding(self.num_unique_categories, dim)
        self.register_buffer('categories_offset', torch.tensor( + list(categories[:-1])).cumsum(dim=0))
        
        self.transformer = TransformerBlock(
            depth=depth,
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        
        if self.num_continuous > 0:
            self.norm = nn.LayerNorm(self.num_continuous)
            self.continuous_mean_std = continuous_mean_std
        else:
            self.norm = nn.Identity()
            self.continuous_mean_std = None

    def forward(self, x_cat, x_cont):
        # Nội suy chỉ số nhúng (embedding index interpolation)
        x_cat = x_cat + self.categories_offset
        x_cat_emb = self.embeds(x_cat)
        
        # Áp dụng cơ chế Multi-Head Self-Attention để trích xuất ngữ cảnh
        x_cat_encoded = self.transformer(x_cat_emb)
        
        # Làm phẳng ma trận đặc trưng phân loại
        flat_cat = rearrange(x_cat_encoded, 'b n d -> b (n d)')
        
        xs = [flat_cat]
        
        # Xử lý đặc trưng liên tục thông qua Layer Normalization
        if self.num_continuous > 0:
            assert x_cont.shape == self.num_continuous, f'Dữ liệu đầu vào phải có chính xác {self.num_continuous} đặc trưng liên tục'
            if self.continuous_mean_std is not None:
                mean, std = self.continuous_mean_std.unbind(dim=-1)
                x_cont = (x_cont - mean) / std
            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)
            
        # Gộp các đặc trưng thành biểu diễn tiềm ẩn cuối cùng
        latent_features = torch.cat(xs, dim=-1)
        
        return latent_features
Cấu trúc trên thiết lập một đường dẫn tính toán nơi các đặc trưng phân loại mạng (như loại giao thức, trạng thái cờ) được biểu diễn thành các vector có số chiều dim=32. Biến categories_offset đóng vai trò duy trì một ma trận nhúng duy nhất nn.Embedding cho toàn bộ các cột phân loại, thay vì phải khởi tạo nhiều ma trận nhúng rời rạc. Quá trình cumsum dịch chuyển chỉ số của từng biến phân loại để chúng không bị trùng lặp không gian địa chỉ khi thực hiện phép tính chiếu (projection). Bằng cách này, thông tin liên quan đến giao thức TCP, UDP hay ICMP được cấp phát một vùng không gian vector tách biệt với trạng thái cờ kết nối (SYN, ACK, FIN), giúp mô hình Transformer tính toán mức độ tương quan chéo (cross-correlation) chính xác thông qua hàm Attention.Lớp TransformerBlock lặp lại 6 lần (depth=6) với 8 đầu chú ý (heads=8), được tinh chỉnh chuyên biệt cho việc phát hiện các tín hiệu nhiễu và mẫu hình tấn công phân tán, một đặc tính vốn rất khó phát hiện nếu chỉ dùng kiến trúc cây quyết định. Đặc biệt, sự xuất hiện của GEGLU (Gated Error Linear Unit) trong lớp FeedForward thay thế cho ReLU truyền thống giúp bảo toàn các gradient nhỏ giọt trong quá trình lan truyền ngược (backpropagation), điều này vô cùng quan trọng khi các bất thường mạng thường chỉ chiếm một phần tỷ lệ siêu nhỏ (dưới 1%) trong tổng thể lưu lượng luồng dữ liệu. Đối với các đặc trưng liên tục (như số lượng byte truyền, thời lượng kết nối), chúng bỏ qua hoàn toàn kiến trúc Transformer và đi thẳng vào lớp chuẩn hóa LayerNorm, giữ nguyên tính chất tuyến tính của mình trước khi được kết hợp bằng hàm torch.cat.Tích hợp Khung Tiền xử lý cho Dữ liệu Phát hiện Xâm nhậpViệc áp dụng kiến trúc trên đòi hỏi một bộ xử lý dữ liệu đầu vào cực kỳ tinh vi. Trong phân tích luồng mạng, hai tập dữ liệu thường được sử dụng nhất để thiết lập chuẩn đánh giá (benchmark) là NSL-KDD và UNSW-NB15. Tập NSL-KDD chứa 41 đặc trưng, trong đó 3 đặc trưng là phân loại (protocol_type, service, flag) và 38 đặc trưng còn lại là liên tục. Tập UNSW-NB15 bao gồm 49 đặc trưng và chứa các lớp tấn công phức tạp như Fuzzers, Exploits, Worms, và Reconnaissance. Do cấu trúc của TabTransformer yêu cầu biết chính xác kích thước không gian từ vựng của mỗi biến phân loại (thông qua tham số categories), đường ống tiền xử lý bằng thư viện Pandas và Scikit-Learn phải mã hóa dữ liệu một cách nhất quán và xuất ra định dạng PyTorch Dataset.Mã nguồn dưới đây xây dựng lớp IDSDataPipeline nhằm đảm bảo việc tiền xử lý không gây rò rỉ dữ liệu (data leakage) giữa tập huấn luyện và tập kiểm thử, đồng thời thiết lập các siêu dữ liệu cần thiết cho TabTransformer.Pythonimport pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE

class IDSTensorDataset(Dataset):
    def __init__(self, x_cat, x_cont, y):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_cont[idx], self.y[idx]

class IDSDataPipeline:
    def __init__(self, cat_cols, cont_cols, target_col):
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.target_col = target_col
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categories_dimensions =
        self.continuous_mean_std = None
        
    def fit_transform(self, df_train, apply_smote=False):
        df = df_train.copy()
        
        # Xử lý giá trị bị thiếu bằng giá trị trung vị và nhãn 'UNKNOWN'
        df[self.cont_cols] = df[self.cont_cols].fillna(df[self.cont_cols].median())
        df[self.cat_cols] = df[self.cat_cols].fillna('UNKNOWN')
        
        # Trích xuất không gian đặc trưng phân loại
        x_cat_list =
        for col in self.cat_cols:
            le = LabelEncoder()
            # Bổ sung token đặc biệt cho dữ liệu chưa từng xuất hiện (Out-of-Vocabulary)
            unique_values = list(df[col].unique()) + ['<OOV>']
            le.fit(unique_values)
            self.label_encoders[col] = le
            self.categories_dimensions.append(len(le.classes_))
            
            encoded_col = le.transform(df[col])
            x_cat_list.append(encoded_col)
            
        x_cat = np.column_stack(x_cat_list)
        
        # Chuẩn hóa đặc trưng liên tục
        x_cont = self.scaler.fit_transform(df[self.cont_cols])
        
        # Lưu trữ giá trị trung bình và độ lệch chuẩn để tích hợp vào TabTransformer
        self.continuous_mean_std = torch.tensor(
            np.stack([self.scaler.mean_, self.scaler.scale_], axis=-1),
            dtype=torch.float32
        )
        
        y = df[self.target_col].values
        
        # Cân bằng dữ liệu (Class Imbalance Handling) sử dụng SMOTE
        if apply_smote:
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            # Yêu cầu gộp các đặc trưng để thực hiện SMOTE, sau đó tách ra
            x_combined = np.hstack((x_cat, x_cont))
            x_resampled, y_resampled = smote.fit_resample(x_combined, y)
            
            # Khôi phục định dạng int cho đặc trưng phân loại sau khi SMOTE nội suy
            x_cat = np.round(x_resampled[:, :len(self.cat_cols)]).astype(int)
            x_cont = x_resampled[:, len(self.cat_cols):]
            y = y_resampled
            
        return x_cat, x_cont, y
        
    def transform(self, df_test):
        df = df_test.copy()
        
        df[self.cont_cols] = df[self.cont_cols].fillna(df[self.cont_cols].median())
        df[self.cat_cols] = df[self.cat_cols].fillna('UNKNOWN')
        
        x_cat_list =
        for col in self.cat_cols:
            le = self.label_encoders[col]
            # Áp dụng token <OOV> cho các giá trị phân loại không nằm trong tập huấn luyện
            mapped_values = df[col].apply(lambda x: x if x in le.classes_ else '<OOV>')
            encoded_col = le.transform(mapped_values)
            x_cat_list.append(encoded_col)
            
        x_cat = np.column_stack(x_cat_list)
        x_cont = self.scaler.transform(df[self.cont_cols])
        y = df[self.target_col].values
        
        return x_cat, x_cont, y
Trong hệ thống IDS mạng, sự xuất hiện của các kết nối zero-day thường mang theo các giao thức hoặc dịch vụ mạng chưa từng được ghi nhận trong tập dữ liệu lịch sử. Cơ chế gán token <OOV> (Out-of-Vocabulary) giải quyết triệt để lỗi ngoại lệ (exception errors) khi tiến hành suy luận (inference) trên môi trường thực tế. Ngoài ra, dữ liệu luồng mạng như UNSW-NB15 bị mất cân bằng nghiêm trọng với hơn 87% gói tin là bình thường (Benign) và chỉ một tỷ lệ nhỏ là các gói tin thám thính (Reconnaissance) hoặc khai thác lỗ hổng (Exploits). Kỹ thuật Oversampling như SMOTE (Synthetic Minority Oversampling Technique) được lồng ghép vào quy trình huấn luyện để khôi phục ranh giới phân loại công bằng, cải thiện mạnh mẽ khả năng phát hiện độ nhạy (Recall) đối với các loại tấn công hiếm. Đặc tính làm tròn np.round sau khi chạy SMOTE là thủ thuật tính toán quan trọng để bảo toàn tính rời rạc của các định danh phân loại trước khi đưa vào lớp Embedding của PyTorch.Trích xuất Đặc trưng Sâu (Deep Feature Extraction) bằng PyTorch TabularMặc dù triển khai thủ công mạng TabTransformer đem lại sự tùy biến cao, việc tích hợp kiến trúc vào một quy trình công nghiệp quy mô lớn có thể được tự động hóa mạnh mẽ thông qua thư viện pytorch_tabular. Khung làm việc này bọc toàn bộ mô hình PyTorch, cấu hình dữ liệu và quy trình huấn luyện Lightning, cho phép kết nối luồng dữ liệu trích xuất với XGBoost một cách gọn gàng bằng lớp DeepFeatureExtractor.Đoạn mã cấu hình luồng PyTorch Tabular cho phép huấn luyện nhúng ngữ cảnh trực tiếp dựa trên hàm mất mát tự giám sát (Self-Supervised Learning) hoặc thông qua giám sát nhị phân, sau đó xuất ra mô hình trích xuất tĩnh (static extractor).Pythonfrom pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.feature_extractor import DeepFeatureExtractor

# Thiết lập cấu hình dữ liệu cho NSL-KDD
data_config = DataConfig(
    target=["label_is_attack"], # Mục tiêu nhị phân (0: Normal, 1: Attack)
    continuous_cols=["duration", "src_bytes", "dst_bytes", "wrong_fragment",...], # 38 đặc trưng
    categorical_cols=["protocol_type", "service", "flag"] # 3 đặc trưng
)

# Cấu hình huấn luyện viên (Trainer) chạy trên GPU
trainer_config = TrainerConfig(
    auto_lr_find=True,
    batch_size=1024,
    max_epochs=50,
    gpus=1, # Sử dụng CUDA acceleration
    early_stopping_patience=10
)

optimizer_config = OptimizerConfig()

# Khởi tạo mô hình TabTransformer với các siêu tham số tiêu chuẩn
model_config = TabTransformerConfig(
    task="classification",
    learning_rate=0.001,
    input_embed_dim=32,
    num_heads=8,
    num_attn_blocks=6,
    transformer_activation="geglu",
    share_embedding=False
)

# Khởi tạo mô hình tabular
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config
)

# Huấn luyện mô hình để học trọng số nhúng (Embedding weights)
# df_train chứa toàn bộ dữ liệu Pandas DataFrame đã làm sạch cơ bản
tabular_model.fit(train=df_train, validation=df_val)

# Khởi tạo DeepFeatureExtractor để tước bỏ lớp phân loại cuối cùng
feature_extractor = DeepFeatureExtractor(tabular_model)

# Biến đổi tập dữ liệu thành các vector tiềm ẩn
X_train_latent = feature_extractor.fit_transform(df_train)
X_test_latent = feature_extractor.transform(df_test)
Sức mạnh của DeepFeatureExtractor nằm ở khả năng truy xuất biểu diễn đặc trưng từ lớp kết nối cuối cùng (penultimate layer) của mạng thần kinh mà không yêu cầu viết lại logic truyền thẳng (forward pass). Sau khi gọi .transform(), đầu ra X_train_latent không còn là tập hợp của 41 cột dữ liệu hỗn hợp (categorical và continuous), mà là một ma trận NumPy thuần nhất dạng số nguyên thực. Kích thước số chiều của ma trận này phụ thuộc vào input_embed_dim (ở đây là 32 nhân với 3 biến phân loại = 96) cộng với 38 biến liên tục ban đầu, tạo ra tổng cộng 134 chiều không gian đặc trưng. Không gian này mang theo các tương tác phi tuyến tính phức tạp mà TabTransformer đã khám phá được, hoàn toàn sẵn sàng cho việc phân tách bằng mặt phẳng siêu việt của cây quyết định.Cấu hình và Tích hợp Thuật toán XGBoost vào Hệ thốngSau khi TabTransformer đã hoàn thành quá trình kiến tạo biểu diễn đa chiều, XGBoost (eXtreme Gradient Boosting) sẽ tiếp nhận ma trận dữ liệu làm đầu vào. XGBoost giải quyết bài toán tối ưu hóa đa mục tiêu bằng cách liên tục xây dựng các cây quyết định (decision trees) nhằm giảm thiểu hàm mất mát (loss function) dựa trên đạo hàm bậc nhất và bậc hai của không gian sai số.Để đảm bảo hiệu năng xử lý hàng triệu bản ghi luồng mạng mà không gây tắc nghẽn bộ nhớ, cấu trúc dữ liệu cơ sở của pandas/NumPy phải được chuyển đổi thành cấu trúc độc quyền xgb.DMatrix của thư viện XGBoost.Pythonimport xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Định dạng dữ liệu thành cấu trúc DMatrix tối ưu của XGBoost
dtrain = xgb.DMatrix(X_train_latent, label=df_train["label_is_attack"].values)
dval = xgb.DMatrix(X_val_latent, label=df_val["label_is_attack"].values)
dtest = xgb.DMatrix(X_test_latent, label=df_test["label_is_attack"].values)

# Thiết lập siêu tham số cho hệ thống IDS phân loại nhị phân
xgb_params = {
    'objective': 'binary:logistic',     # Mục tiêu hồi quy logistic cho dữ liệu 0/1
    'eval_metric': ['logloss', 'auc'],  # Đo lường sai số và diện tích dưới đường cong
    'tree_method': 'gpu_hist',          # Kích hoạt tính toán dựa trên biểu đồ trên NVIDIA GPU
    'learning_rate': 0.05,              # Bước tiến học (eta)
    'max_depth': 6,                     # Độ sâu tối đa của cây để ngăn quá khớp
    'min_child_weight': 1,              # Ngưỡng tổng trọng số tối thiểu để phân tách node
    'subsample': 0.8,                   # Tỷ lệ lấy mẫu dữ liệu cho từng cây
    'colsample_bytree': 0.8,            # Tỷ lệ lấy mẫu đặc trưng cho từng cây
    'alpha': 0.1,                       # Trọng số chuẩn hóa L1 (Lasso)
    'lambda': 1.0,                      # Trọng số chuẩn hóa L2 (Ridge)
    'seed': 42
}

# Danh sách tập đánh giá để theo dõi quá trình hội tụ
eval_list = [(dtrain, 'train'), (dval, 'eval')]

# Bắt đầu quá trình huấn luyện Boosting
xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=1000,               # Số lượng cây tối đa
    evals=eval_list,
    early_stopping_rounds=50,           # Dừng nếu sai số trên tập eval không giảm sau 50 vòng
    verbose_eval=100
)

# Tiến hành suy luận trên tập kiểm thử
y_pred_prob = xgb_model.predict(dtest)
# Áp dụng ngưỡng cắt (threshold) 0.5 để quyết định nhãn tấn công
y_pred_class = (y_pred_prob >= 0.5).astype(int)
y_true = df_test["label_is_attack"].values

# Đánh giá hiệu năng mạng lưới IDS
print("=== Kết quả Kiểm thử Hệ thống IDS ===")
print(f"Accuracy:  {accuracy_score(y_true, y_pred_class):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_class):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred_class):.4f}")
print(f"F1-Score:  {f1_score(y_true, y_pred_class):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_class))
Trong đoạn mã trên, tham số tree_method: 'gpu_hist' được sử dụng để chuyển đổi quy trình tìm kiếm ranh giới phân chia (split finding) từ quét toàn bộ dữ liệu (exact greedy) sang sử dụng biểu đồ khối cục bộ (histogram-based) vận hành trực tiếp trên VRAM của GPU. Sự cải tiến này là bắt buộc trong môi trường thực tế, cho phép mô hình IDS phân tích hàng gigabyte log dữ liệu (như CIC-IDS-2017) với tốc độ tăng tốc gấp hàng chục lần so với thiết lập CPU truyền thống. Hơn nữa, việc cấu hình song song subsample: 0.8 và colsample_bytree: 0.8 giới thiệu sự ngẫu nhiên ngẫu thức (stochastic randomness) vào thuật toán, hoạt động như một tầng bảo vệ ngăn chặn việc các cây quyết định ghi nhớ thụ động (memorize) các mảng đặc trưng cục bộ do TabTransformer tạo ra. Sự kết hợp của hàm mục tiêu binary:logistic cung cấp xác suất độ tin cậy của mối đe dọa (từ 0.0 đến 1.0), tạo cơ sở để quản trị viên mạng điều chỉnh ngưỡng cắt (threshold), từ đó kiểm soát chặt chẽ bài toán đánh đổi giữa Cảnh báo giả (False Positives) và Bỏ sót tấn công (False Negatives). Nếu áp dụng vào cấu hình UNSW-NB15 để phân loại chi tiết 9 dạng tấn công, siêu tham số sẽ được cập nhật thành objective: 'multi:softmax' cùng với num_class: 10.Tối ưu hóa Siêu tham số Bằng Optuna (Hyperparameter Optimization)Kiến trúc liên hợp TabTransformer và XGBoost chứa một tập hợp lớn các siêu tham số. Thay vì thử nghiệm mù quáng bằng tay (manual trial-and-error), việc sử dụng khung làm việc tìm kiếm tự động Bayesian như Optuna đảm bảo khai thác tối đa năng lực của mô hình trên các bề mặt tổn thất phức tạp. Quá trình này thiết lập một hàm mục tiêu (objective function) nhắm vào việc tối đa hóa chỉ số F1-Score hoặc Recall, đặc biệt thiết yếu trong IDS để đảm bảo phát hiện được mọi nỗ lực xâm nhập.Sự phân bổ biên độ tối ưu cho các tham số TabTransformer và XGBoost được định hình chi tiết thông qua các lưới tìm kiếm (grid space) thực tế :Kiến trúcSiêu Tham sốGiải thích Vai trò trong Phân tích IDSPhạm vi Tối ưu (Optuna Bounds)Giá trị Điển hìnhTabTransformerdim (Kích thước nhúng)Quyết định chiều rộng không gian ngữ nghĩa đại diện cho các thuộc tính mạng như IP hoặc cổng. Kích thước quá lớn gây phân tán ma trận, quá nhỏ làm mất thông tin.32TabTransformerdepth (Số khối tự chú ý)Kiểm soát số lần mô hình xem xét lại quan hệ chéo giữa các giao thức và trạng thái TCP.6TabTransformerheads (Số đầu chú ý)Cho phép mạng phân bổ sự chú ý song song đến các vector tấn công khác nhau ở cùng một chu kỳ thời gian.8TabTransformerattn_dropoutCắt đứt các liên kết giả mạo giữa các luồng mạng bình thường và nhiễu ngẫu nhiên.[0.1, 0.2, 0.3, 0.4]0.1XGBoostmax_depthGiới hạn số lượng truy vấn tuần tự mà cây quyết định tạo ra. Mạng IDS cần độ sâu vừa đủ để nhận diện tấn công nhiều bước.6XGBoostlearning_rate ($\eta$)Tốc độ mà thuật toán XGBoost sử dụng để bù đắp sai số cho các cây trước đó.[0.01, 0.05, 0.1, 0.2]0.05XGBoostn_estimatorsTổng số lượng cây con được tạo ra. Được theo dõi cùng với cơ chế Early Stopping để ngăn chặn dư thừa tài nguyên.500Đoạn mã cấu hình vòng lặp Optuna để kết nối cả hai kiến trúc diễn ra như sau:Pythonimport optuna
from sklearn.metrics import recall_score

def objective(trial):
    # Định nghĩa không gian tìm kiếm cho XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True)
    }
    
    # Do quá trình train TabTransformer tốn nhiều thời gian, ta giả định biểu diễn
    # latent X_train_latent đã được sinh ra trước với (dim=32, depth=6, heads=8)
    
    dtrain = xgb.DMatrix(X_train_latent, label=y_train)
    dval = xgb.DMatrix(X_val_latent, label=y_val)
    
    bst = xgb.train(
        params, 
        dtrain, 
        num_boost_round=500, 
        evals=[(dval, 'eval')], 
        early_stopping_rounds=30, 
        verbose_eval=False
    )
    
    y_pred_prob = bst.predict(dval)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Mục tiêu tối thượng của IDS là giảm thiểu âm tính giả (False Negatives)
    # nên ta trả về Recall để Optuna tối đa hóa
    recall = recall_score(y_val, y_pred)
    return recall

# Khởi chạy chu trình tối ưu hóa
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best XGBoost Hyperparameters:", study.best_params)
print("Highest Validation Recall:", study.best_value)
Kiến trúc tìm kiếm Bayesian này tự động gạt bỏ các khu vực siêu tham số không hứa hẹn trong quá trình tìm kiếm, giúp kỹ sư an ninh mạng tiết kiệm hàng tuần tính toán GPU trên những bộ dữ liệu khổng lồ như CIC-IDS-2017.Diễn giải Khả năng Ra quyết định Mạng với SHAPSự hoài nghi lớn nhất của các tổ chức bảo mật đối với các mô hình mạng nơ-ron (kể cả Transformer) nằm ở bản chất "hộp đen" (black-box) của chúng. Trong môi trường điều tra số (Digital Forensics) và phản ứng sự cố (Incident Response), nếu hệ thống gắn cờ một địa chỉ IP là nguồn tấn công DDoS, nó phải cung cấp lý do kỹ thuật chi tiết. Bằng việc chuyển đổi lớp đưa ra quyết định cuối cùng từ MLP sang XGBoost, toàn bộ chuỗi quyết định của mô hình bỗng chốc trở nên có thể diễn giải được bằng toán học thông qua giá trị SHAP (SHapley Additive exPlanations).SHAP tiếp cận bài toán bằng lý thuyết trò chơi hợp tác (cooperative game theory), gán cho mỗi đặc trưng đầu vào một giá trị đại diện cho tầm ảnh hưởng biên (marginal contribution) của nó vào quyết định cuối cùng. Do XGBoost là một tập hợp các cây quyết định, lớp shap.TreeExplainer được tích hợp cung cấp tốc độ tính toán giá trị SHAP nội hàm cực kỳ nhanh chóng thay vì phải chạy giải thuật hoán vị mô phỏng (permutation) chậm chạp.Pythonimport shap
import matplotlib.pyplot as plt

# Thiết lập bộ giải thích dạng cây trên mô hình XGBoost đã huấn luyện
explainer = shap.TreeExplainer(xgb_model)

# Tính toán ma trận SHAP cho toàn bộ tập kiểm thử
shap_values = explainer.shap_values(dtest)

# Vẽ biểu đồ tổng quan tầm quan trọng của đặc trưng (Summary Plot)
# Cần lưu ý X_test_latent là mảng không tên, do đó cần cung cấp danh sách 
# tên các đặc trưng (gồm biến phân loại nhúng + biến liên tục) để hiển thị
feature_names = [f"Embedded_Cat_{i}" for i in range(num_embedded_features)] + list(cont_cols)

shap.summary_plot(
    shap_values, 
    X_test_latent, 
    feature_names=feature_names, 
    show=False
)
plt.title("Biểu đồ Tầm quan trọng Đặc trưng SHAP đối với IDS")
plt.tight_layout()
plt.savefig("shap_summary_ids.png")
Kết quả từ biểu đồ SHAP chỉ ra rằng, đối với lớp dữ liệu bất thường (Anomalies), các biến số liên tục đại diện cho kích thước luồng (như src_bytes, dst_bytes trong NSL-KDD) và các đặc trưng ngữ cảnh nhúng từ TabTransformer đại diện cho loại hình dịch vụ (service như http, ftp_data) có mức độ đóng góp quan trọng nhất. Sự hiện diện của các giá trị SHAP âm cho thấy khi các gói tin mang đặc tính của dịch vụ truyền thống ổn định, mô hình có xu hướng đẩy xác suất tấn công về 0 (Benign). Cơ chế diễn giải này không chỉ chứng minh tính chính xác của mô hình mà còn là cơ sở để phát triển các quy tắc tường lửa (Firewall Rules) và chữ ký tấn công (Attack Signatures) mới.Đánh giá tổng quát hiệu năng khi chạy trực tiếp đường ống TabTransformer + XGBoost trên tập NSL-KDD đem lại các số liệu đột phá. Trong khi mô hình XGBoost thuần túy chỉ đạt mức Recall vào khoảng 95.8%, và TabTransformer theo dạng End-to-End (sử dụng MLP) đạt 96.50% , mô hình kết hợp trực tiếp nâng cao độ chính xác Recall lên ngưỡng vượt trội 99.27%.Kiến trúc Phân loạiĐộ Chính xác (Accuracy)F1-ScoreRecallKhả năng Giải thích (Interpretability)XGBoost Thuần túy96.50%96.45%95.80%Tốt (TreeExplainer)TabTransformer Thuần túy97.50%97.22%96.50%Thấp (Black-box MLP)TabTransformer + XGBoost99.27%99.15%99.27%Tốt (Latent SHAP Tracing)Thống kê này khẳng định sức mạnh của đặc trưng ngữ cảnh sâu. Mô hình đã tự thân học được các khuôn mẫu lẩn tránh tinh vi của tấn công thăm dò (Probe) hoặc tấn công leo thang đặc quyền (U2R), vốn là những dạng thức lẩn trốn sự truy xuất của các quy tắc phi tuyến tính đơn thuần của GBDT.Triển khai Hệ thống và Khuyến nghị Xây dựng API (Production & Deployment)Một khi được chứng minh tính hiệu quả, mã nguồn cần được cấu trúc lại để phù hợp với quy trình MLOps trên môi trường đám mây và hệ thống mạng cục bộ (On-premise SOC - Security Operations Center). Trong thực tế sản xuất, kiến trúc này thường được chứa (containerized) thông qua Docker và đưa lên các nền tảng điều phối như Amazon SageMaker để vận hành liên tục.SageMaker cung cấp sẵn các giao diện xử lý mô hình XGBoost (XGBoost Framework Processor) và tính năng triển khai điểm cuối (Endpoint Deployment) với khả năng nhận luồng suy luận trực tuyến. Quá trình này tận dụng việc biên dịch chéo (cross-compilation) lớp PyTorch TabTransformer sang định dạng mã hóa tốc độ cao như ONNX hoặc TorchScript, nhằm đảm bảo thời gian trễ (latency) khi trích xuất vector nhúng gói tin duy trì ở mức dưới 5 phần nghìn giây (5ms).Kiến trúc triển khai qua ứng dụng HTTP (FastAPI hoặc Flask) được thiết lập nhằm xây dựng một webhook lắng nghe sự kiện từ trình phân tích gói tin (ví dụ: Zeek hoặc Suricata) và lập tức trả về nhãn nhận diện.Pythonfrom flask import Flask, request, jsonify
import torch
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Tải cấu trúc mô hình đã được huấn luyện vào RAM
# Giả định tab_extractor và xgb_model đã được lưu trên đĩa
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tab_extractor = torch.load('tabtransformer_extractor.pth').to(device)
tab_extractor.eval()

xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_ids_model.json')

@app.route('/predict_flow', methods=)
def predict_flow():
    try:
        # Nhận dữ liệu gói tin luồng (network flow data) định dạng JSON
        data = request.json
        x_cat_raw = np.array(data['categorical_features'])
        x_cont_raw = np.array(data['continuous_features'])
        
        # Tiền xử lý nhanh: Chuyển sang Tensor
        x_cat = torch.tensor(x_cat_raw, dtype=torch.long).unsqueeze(0).to(device)
        x_cont = torch.tensor(x_cont_raw, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Giai đoạn 1: Biến đổi qua TabTransformer để lấy vector tiềm ẩn
        with torch.no_grad():
            latent_features = tab_extractor(x_cat, x_cont)
            
        latent_np = latent_features.cpu().numpy()
        
        # Giai đoạn 2: Phân loại bằng XGBoost
        dtest = xgb.DMatrix(latent_np)
        pred_prob = xgb_model.predict(dtest)
        
        # Ngưỡng phân loại phát hiện xâm nhập
        is_attack = bool(pred_prob >= 0.5)
        
        return jsonify({
            'status': 'success',
            'is_attack': is_attack,
            'threat_probability': float(pred_prob)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Chạy máy chủ gunicorn đa luồng cho môi trường sản xuất
    app.run(host='0.0.0.0', port=5000, threaded=True)
Giao diện webhook trên chứng minh khả năng di động tuyệt vời của giải pháp. Nhờ việc phân định rạch ròi quy trình trích xuất đặc trưng mạng nơ-ron bằng PyTorch và xử lý luận lý bằng công cụ C++ của XGBoost, kiến trúc không gặp hiện tượng thắt cổ chai tính toán (computational bottleneck). Để đáp ứng tốc độ xử lý băng thông mạng lên đến hàng chục Gigabit/giây (Gbps), các kỹ sư an toàn thông tin được khuyến nghị cấu trúc lại đầu vào thành cơ chế gom lô nhỏ (Mini-batching streaming buffer) thay vì truyền từng gói tin đơn lẻ, qua đó khai thác song song toàn bộ hàng nghìn lõi CUDA trên GPU NVIDIA.Quy trình thiết kế này không chỉ đặt ra tiêu chuẩn mới cho nền tảng IDS hiện đại mà còn xác lập một quy chuẩn kỹ thuật trong kỹ nghệ dữ liệu bảo mật: Bất kỳ khi nào dữ liệu mạng có chứa số lượng lớn biến phân loại (địa chỉ MAC, IP, giao thức) xen kẽ với dữ liệu định lượng (kích thước, thời gian), phương pháp kết hợp mô hình học biểu diễn ngữ cảnh với nền tảng phân loại Boosting sẽ luôn cho ra hiệu năng nhận diện và khả năng diễn giải mạnh mẽ và tối ưu nhất.