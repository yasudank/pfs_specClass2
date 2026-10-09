# Subaru PFS Spectral Classification Pipeline

**2-Channel SimCLR (Channel 1 Boost $\alpha=3.0$) + Latent SMOTE (3x) Random Forest Classifier**

本リポジトリは、すばる望遠鏡 主焦点分光器（Subaru Prime Focus Spectrograph; PFS）で取得された 216,000 天体超の大規模 1次元スペクトルデータ（波長範囲 370 nm 〜 1260 nm）を対象とし、自己教師あり対照学習（SimCLR）と不均衡データ緩和技術（SMOTE）を統合した **高精度・物理的解釈可能性を備えたスペクトル分類パイプライン** です。

---

## 1. 概要と背景

PFS サーベイにおける天体分類（GALAXY / STAR / QSO）では、**QSO（クエーサー）の出現頻度がサーベイ全体の約 0.6% 〜 3% に過ぎないという極端なクラス不均衡（Extreme Class Imbalance）** が生じます。
従来の単一チャネルモデルや通常の機械学習手法では、以下の課題により QSO の見逃し（Low Recall）が頻発していました：
1. **母銀河星光（Host Galaxy Starlight）による埋没**:
   低〜中赤方偏移（$z \lesssim 1.5$）の AGN では、母銀河の連続光や 4000Å break が支配的となり、モデルが銀河の連続光形状（SED）に過剰適合して銀河と誤認する。
2. **局所輝線シグナルの希釈**:
   広輝線（Mg II, H$\alpha$, C IV 等）のエネルギー寄与が全体の連続光に比べて小さいため、単一チャネル入力ではエンコーダが輝線の存在を十分に表現できない。

本パイプラインでは、**「2チャネル分解 ＋ 輝線残差の振幅ブースト（$\alpha = 3.0$）」** と **「潜在空間での SMOTE 3倍水増し」** を導入することで、QSO 再現率を **42.7% から 67.3% へと大幅に向上（+24.6%）** させ、サーベイ全体で 98.05% の高精度分類を達成しました。

---

## 2. アルゴリズムとパイプライン設計

パイプラインは以下の 4 ステップから構成されます。

```
[Raw PFS FITS Spectra (370-1260nm)]
               │
               ▼
[Step 1: 前処理 & 2チャネル分解]
   Channel 0: 観測連続光 F(λ)
   Channel 1: 局所輝線残差 3.0 × R(λ)  [α=3.0 ブースト]
               │
               ▼
[Step 2: 自己教師あり 2-Channel SimCLR 学習 (GPU: RTX 4090)]
   - 1D ResNet / ConvNet エンコーダ
   - 天文学的スペクトル拡張 (マスキング, 連続光傾斜, ノイズ)
   - NT-Xent (InfoNCE) 対照損失 (τ=0.10)
               │
               ▼
[Step 3: 128次元 潜在表現抽出 (Survey-wide Embeddings)]
   - 各天体の 128D ベクトル表現 z ∈ R^128 を抽出
               │
               ▼
[Step 4: 潜在空間 SMOTE (3x) + Random Forest 分類]
   - 少数派 QSO の 128D 表現を k-NN 補間 (SMOTE 3倍)
   - 5-Fold Stratified Cross-Validation (balanced_subsample)
               │
               ▼
[最終出力: 予測クラス (OOF), 信頼度確率, 潜在診断図, スタックスペクトル]
```

### 2.1 2チャネル分解と輝線振幅ブースト ($\alpha = 3.0$)
各スペクトル $F(\lambda)$ に対し、平滑化連続光 $C(\lambda)$（Median Filter window=128 + Gaussian filter $\sigma=32$）を差し引いた局所輝線成分 $R(\lambda) = F(\lambda) - C(\lambda)$ を分離します。
局所輝線チャネルの振幅を $\alpha = 3.0$ 倍にブーストした 2 チャネルテンソルを入力とします：
$$\mathbf{X}(\lambda) = \begin{bmatrix} F(\lambda) \\ 3.0 \cdot R(\lambda) \end{bmatrix} \in \mathbb{R}^{2 \times 4096}$$
* **効果**: エンコーダが全体的な連続光（Channel 0）のみに引きずられることを防ぎ、微弱な広輝線・狭輝線（Channel 1）に高い注意を払う潜在表現を学習します。

### 2.2 自己教師あり対照学習（2-Channel SimCLR）
* **ネットワーク**: 1D ResNet / 畳み込みバックボーン ($C_{\mathrm{in}}=2$) ＋ 128次元潜在表現層 ＋ 64次元非線形射影ヘッド。
* **データ拡張（SpectralAugmentation）**:
  - Sky ライン・検出器ギャップを模擬したランダムマスキング（幅 48 px, 3 箇所, 確率 60%）
  - 連続光の微小傾斜（Tilt $\pm 8\%$）
  - 波長シフト（$\pm 512$ px）
  - ガウシアンノイズ付加（$\sigma = 0.02$）
* **損失関数**: NT-Xent (InfoNCE) Contrastive Loss（温度パラメータ $\tau = 0.10$）。

### 2.3 潜在空間 SMOTE (3x) 水増し ＋ Random Forest
* 抽出された 128 次元潜在空間において、QSO の訓練サンプルに対して $k=5$ 最近傍法に基づく SMOTE（Synthetic Minority Over-sampling Technique）を適用し、潜在空間上で QSO ベクトルを 3 倍に水増し。
* 分類器: `RandomForestClassifier` (n_estimators=100, max_depth=16, class_weight='balanced_subsample')。
* 5-Fold Stratified Cross-Validation による Out-of-Fold (OOF) 評価。

---

## 3. 性能評価結果（Benchmark Comparison）

サーベイ代表データセット（40,544 天体）における 5-Fold CV の分類精度比較：

| モデル設定 | QSO Recall | QSO Precision | QSO F1-Score | 全体 Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (単一チャネル SimCLR + 通常 RF)** | 42.70% | **92.86%** | 0.5851 | 97.46% | 0.8540 |
| **2-Channel Standard ($\alpha=1.0$ + 通常 RF)** | 41.48% | 86.84% | 0.5615 | 97.35% | 0.8447 |
| **実験1 (2-Channel $\alpha=1.0$ + SMOTE 3x RF)** | 55.61% | 84.14% | 0.6694 | 97.74% | 0.8761 |
| **実験2 (2-Channel $\alpha=3.0$ Boost + 通常 RF)** | 53.63% | 85.51% | 0.6588 | 97.68% | 0.8715 |
| **★ 最終結果 (2-Channel $\alpha=3.0$ + SMOTE 3x RF)** | **67.30%** | **85.70%** | **0.7540** | **98.05%** | **0.9099** |

### 重要な成果
1. **QSO 検出数の大幅増**: 真の QSO 1,309 天体のうち **881 天体を正しく同定**（Baseline から +322 天体増加）。
2. **新規 QSO 候補の発見**: DB カタログで GALAXY とされていた天体の中から、モデルが強い確信度で QSO と予測した **116 天体の New QSO Candidates** を同定。静止系スタックスペクトルにより、明瞭な広輝線（Mg II, C IV）を持つ本物のクエーサーであることを確認。
3. **未検出 QSO の物理的究明**: 依然として見逃されている 303 天体（Remaining Missed QSOs $\to$ GALAXY）について 128D コサイン類似度を解析した結果、対銀河類似度が $\mathrm{Sim}_{\mathrm{GAL}} = 0.768 > \mathrm{Sim}_{\mathrm{QSO}} = 0.714$ と銀河領域内部に深く位置しており、母銀河の星光連続光（4000Å break 等）の寄与が支配的であることが実証されました。

---

## 4. リポジトリ構成とファイル説明

```
pfs_specClass2/
├── README.md                                          # 本ドキュメント
├── environment.yml                                    # 本プロジェクト専用 Conda 仮想環境定義
├── requirements.txt                                   # Pip 依存パッケージ一覧
├── setup_env.sh                                       # 専用仮想環境 (pfs_specclass) 自動構築スクリプト
├── run_pipeline.sh                                    # 全パイプライン一括実行シェルスクリプト
├── config.py                                          # 波長グリッド、アームマスク、パス設定
├── dataset.py                                         # FITS読み込み・前処理・標準化
├── augmentations.py                                   # 2チャネル対照学習用スペクトルデータ拡張
│
├── models/
│   └── contrastive_model.py                           # 2-Channel 1D CNN SimCLR & NTXentLoss
│
├── step1_prepare_full_augmented_dataset.py            # [Step 1] 生FITSデータからデータセット構築
├── train_2channel_boosted_simclr.py                   # [Step 2] α=3.0 ブースト SimCLR 学習 & 128D 抽出
├── run_experiment3_full_analysis.py                   # [Step 3] SMOTE 3x RF 5-Fold CV & 診断図作成
│
├── visualize_remaining_missed_qsos_latent.py          # [Step 4-A] 残余見逃しQSOの潜在空間位置解析
├── plot_remaining_missed_qso_spectra.py               # [Step 4-B] 残余見逃しQSOの個別スペクトル描画
├── plot_individual_missed_qsos_latent_positions.py    # [Step 4-C] 個別天体の潜在空間プロット
│
├── create_216k_full_survey_predictions_catalog.py     # [Step 5] 全216k天体 最終予測カタログ作成 (CSV, SQLite)
│
└── results/                                           # 出力成果物ディレクトリ
    ├── pfs_full_survey_predictions.sqlite3            # 【最終カタログ】全216,476天体 予測DB (主キー付)
    ├── pfs_full_survey_predictions.csv.gz             # 【最終カタログ】全216,476天体 予測CSV (gzip)
    ├── pfs_full_survey_predictions.csv                # 【最終カタログ】全216,476天体 予測CSV
    ├── spectral_simclr_2channel_boost3x.pt            # 学習済み SimCLR 重みファイル
    ├── embeddings_representative_boost3x.npy          # 128次元 潜在表現ベクトル (N=40,544)
    ├── rf_oof_boost3x_smote_representative.npz        # RF 予測ラベル・予測確率・真値
    ├── latent_distribution_boost3x_smote.png          # 潜在空間 t-SNE 分布図 (4パネル)
    ├── rf_disagreement_restframe_spectra_boost3x.png  # 静止系メディアンスタックスペクトル (5パネル)
    ├── remaining_missed_qsos_latent_positions.png     # 残余見逃しQSOの潜在類似度診断図 (4パネル)
    ├── remaining_missed_qso_sample_spectra.png        # 代表見逃しQSOの観測/静止系スペクトル
    └── individual_missed_qsos_latent_positions.png    # 代表天体の潜在パラメータ空間プロット
```

---

## 5. 仮想環境の構築と実行方法（Environment Setup & Run）

### 5.1 プロジェクト専用仮想環境（`pfs_specclass`）の構築

本プロジェクトでは、外部パイプライン（LSST scipipe 等）への依存を排除し、再現性と軽量性を確保するため **専用の仮想環境 `pfs_specclass`** を構築・利用します。

* **Python バージョン**: Python 3.10
* **主要ライブラリ**: PyTorch (CUDA 12+ 対応), Astropy, Scikit-Learn, SciPy, NumPy, Matplotlib, Pandas, Joblib
* **推奨 GPU**: NVIDIA RTX 4090 (24GB VRAM) または同等以上の GPU

#### ① 自動セットアップ（推奨）
付属の [`setup_env.sh`](file:///mnt/ugnas/work/PFS/pfs_specClass2/setup_env.sh) を実行することで、専用仮想環境の作成から GPU 有効化まで自動で完了します：
```bash
cd /mnt/ugnas/work/PFS/pfs_specClass2
./setup_env.sh
```

#### ② 手動セットアップ（Conda / Mamba）
```bash
# environment.yml から専用仮想環境を作成
conda env create -f environment.yml

# PyTorch (CUDA 12+) のインストール
conda activate pfs_specclass
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### ③ 仮想環境のアクティベートと動作確認
```bash
conda activate pfs_specclass

# GPU (RTX 4090) および主要ライブラリの動作確認
python -c "import torch, astropy, sklearn; print('PyTorch CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

---

### 5.2 ワンコマンド一括実行
仮想環境をアクティベートした状態で、または未アクティベートの状態でも、`run_pipeline.sh` を実行するだけで環境を自動検知して全パイプラインを実行します：

```bash
cd /mnt/ugnas/work/PFS/pfs_specClass2
./run_pipeline.sh
```

---

### 5.3 個別ステップの実行

#### 【Step 1】生データからデータセットを作成
FITS ファイルからスペクトルを読み込み、共通波長グリッド（4096 px）上に内挿・規格化してキャッシュします：
```bash
python step1_prepare_full_augmented_dataset.py
```
* **出力**: `results/dataset_full_augmented_scratch.npz`

#### 【Step 2】α=3.0 ブースト 2-Channel SimCLR の学習と特徴量抽出
Channel 1 に 3倍の振幅ブーストを適用して 15 エポックの自己教師あり学習を行い、128 次元潜在表現を抽出します：
```bash
python train_2channel_boosted_simclr.py
```
* **出力**:
  - `results/spectral_simclr_2channel_boost3x.pt`（学習済みモデル）
  - `results/embeddings_representative_boost3x.npy`（128D ベクトル）

#### 【Step 3】潜在空間 SMOTE 3x 水増し ＋ Random Forest 5分割交差検証
抽出された潜在特徴量に対して SMOTE 3倍水増しを適用し、ランダムフォレストの 5-Fold CV を実行します：
```bash
python run_experiment3_full_analysis.py
```
* **出力**:
  - `results/rf_oof_boost3x_smote_representative.npz`（Out-of-Fold 予測結果）
  - `results/latent_distribution_boost3x_smote.png`（潜在空間分布図）
  - `results/rf_disagreement_restframe_spectra_boost3x.png`（静止系スタックスペクトル）

#### 【Step 4】潜在パラメータ位置と個別スペクトルの詳細診断
見逃された天体（Remaining Missed QSOs $\to$ GALAXY）の振る舞いを物理的・幾何学的に検証するスクリプト群です：
```bash
# 1. 128D コサイン類似度・決定境界・類似度マージン分布の解析
python visualize_remaining_missed_qsos_latent.py

# 2. 代表的な見逃し天体の観測系・静止系個別スペクトルの描画
python plot_remaining_missed_qso_spectra.py

# 3. 個別天体を潜在パラメータ空間（t-SNE / PCA / 類似度空間）上にマッピング
python plot_individual_missed_qsos_latent_positions.py
```
* **出力**:
  - `results/remaining_missed_qsos_latent_positions.png`
  - `results/remaining_missed_qso_sample_spectra.png`
  - `results/individual_missed_qsos_latent_positions.png`

#### 【Step 5】全 216k 天体の最終予測カタログ生成
モデルの予測結果（天体種別・信頼度確率）と赤方偏移推定値を統合し、サーベイ全天体の一意な複合主キー `(catId, objId, combination)` を持つ最終カタログを出力します：
```bash
python create_216k_full_survey_predictions_catalog.py
```
* **出力**:
  - `results/pfs_full_survey_predictions.sqlite3`（SQLite データベース）
  - `results/pfs_full_survey_predictions.csv.gz`（圧縮 CSV カタログ）
  - `results/pfs_full_survey_predictions.csv`（非圧縮 CSV カタログ）

---

## 6. 赤方偏移（Redshift）および視線速度の推定・物理検証

本パイプラインでは、天体種別分類（Stage 1: GALAXY / STAR / QSO）に続き、各種別に最適化された赤方偏移・運動学パラメーター推定（Stage 2）を実施します（※これらの推定手法はベンチマーク検証を経て `create_216k_full_survey_predictions_catalog.py` に統合されています）。

### 6.1 銀河（GALAXY）の赤方偏移推定: 1D Deep ResNet
* **手法**: 観測スペクトル（4096次元）を直接入力とする 1D Deep Residual Network（Conv1d + Residual Blocks + Fully Connected 回帰層）により、連続光と主要な吸収線・輝線（[O II], Hβ, [O III], Hα など）の大域的パターンから赤方偏移 $z$ を直接予測。
* **精度評価（テストセット）**:
  - Normalized Median Absolute Deviation ($\sigma_{\mathrm{NMAD}}$): **0.0136**
  - バイアス（Median Error）: **+0.0014**
  - 破局的誤差率（$|\Delta z| / (1+z) > 0.05$）: **12.7%**

### 6.2 クエーサー（QSO）の赤方偏移推定: Two-Stage Hybrid
クエーサーは広い赤方偏移範囲（$z \sim 0.1 - 4.5$）に分布し、Lyα, C IV, C III], Mg II などの広輝線の非対称性やピークシフトが生じます。これに対応するため、深層学習と高精度物理テンプレート相互相関（Cross-Correlation Function; CCF）を組み合わせたハイブリッド手法を採用しました。
* **Stage 1 (Coarse Estimation)**: 1D ResNet 回帰により、多峰性ピークの誤認を防ぎ大まかな赤方偏移 $z_{\mathrm{coarse}}$ を同定。
* **Stage 2 (Fine Estimation)**: $z_{\mathrm{coarse}} \pm 0.15$ の探索窓において、PFS 合成 QSO テンプレートとの相互相関（グリッド刻み $\Delta z = 0.0005$）を計算。検出器のアーム間ギャップ波長域に重みゼロのマスクを適用し、偽の相関ピークを完全排除。
* **精度評価（テストセット）**:
  - $\sigma_{\mathrm{NMAD}}$: **0.0060**（高分解能分光と同等の極めて高い精度）
  - 破局的誤差率: **11.01%**
  - バイアス: **+0.0002**

### 6.3 恒星（STAR）の視線速度（Radial Velocity）測定の検証
恒星スペクトルの微小ドップラーシフト（視線速度 $v_{\mathrm{rad}}$）について、銀河・QSO と同様の大域的テンプレート CCF 手法で測定可能かを検証しました。
* **検証内容**:
  1. データベース上の `target_summary.bestVelocity`（m/s）と CCF 推定速度の比較。
  2. DB 視線速度を用いて rest-frame にデシフトした高精度恒星テンプレートを作成し、吸収線（Mg I, Na I, Ca II トリオレット等）の深度が 20%〜29% 先鋭化することを確認。
* **結論と物理的制約**:
  - PFS のリサンプル後ピクセル分解能は $\approx 89.7\text{ km/s/pixel}$ であり、大域的 CCF のピーク探索刻み（数 km/s）がピクセルスケールより著しく微細な領域に入る。
  - 恒星の多様なスペクトル型（有効温度 $T_{\mathrm{eff}}$、表面重力 $\log g$、金属量 $[\mathrm{Fe}/\mathrm{H}]$）によるテンプレートミスマッチが存在する場合、数十 km/s オーダーの視線速度を単一合成テンプレート CCF で精密決定することは困難。
  - より高精度な視線速度測定には、SDSS/ELODIE などの高分解能多次元グリッドテンプレートフィッティングまたは狭帯域吸収線重心測定が必要であると結論付け、全天体カタログでは恒星の赤方偏移を静止系基準の `est_redshift = 0.0` と設定しています。
*(※ アルゴリズム選定・性能評価に用いた個別テスト・検証スクリプト群 `test_*.py`, `evaluate_*.py` はローカル環境に保存・アーカイブされています)*

---

## 7. 全天体予測カタログの生成とデータベース仕様（Survey Predictions Catalog）

### 7.1 カタログ概要
Subaru PFS サーベイで観測された **全 216,476 スペクトル** に対し、Stage 1（天体種別分類・各クラス確率）および Stage 2（銀河・クエーサーの赤方偏移推定）を実行した最終予測カタログです。
データベースの一意性制約（Composite Primary Key）として **`(catId, objId, combination)`** を採用し、DB から容易に取得可能な情報（真のクラスや真の赤方偏移など）を除外した、純粋なパイプライン予測値のみを格納したクリーンなスキーマとなっています。

* **総レコード数**: 216,476 件
* **予測クラス内訳**:
  - `GALAXY`: 105,420 天体（48.70%）
  - `STAR`: 109,872 天体（50.75%）
  - `QSO`: 1,184 天体（0.55%）
    - ※ DB 上で QSO とラベルされた天体との一致数（Agreed QSO）は **904 天体** を記録し、極端な不均衡下でも SMOTE 拡張モデルの優れた QSO 感度を完全維持。

### 7.2 カタログファイル
* **SQLite データベース**: [`results/pfs_full_survey_predictions.sqlite3`](file:///mnt/ugnas/work/PFS/pfs_specClass2/results/pfs_full_survey_predictions.sqlite3)
* **圧縮 CSV カタログ**: [`results/pfs_full_survey_predictions.csv.gz`](file:///mnt/ugnas/work/PFS/pfs_specClass2/results/pfs_full_survey_predictions.csv.gz)（約 5.0 MB）
* **非圧縮 CSV カタログ**: [`results/pfs_full_survey_predictions.csv`](file:///mnt/ugnas/work/PFS/pfs_specClass2/results/pfs_full_survey_predictions.csv)（約 20 MB）

### 7.3 カタログスキーマ（Table: `survey_predictions`）

| カラム名 | データ型 | 制約 | 説明 |
| :--- | :---: | :---: | :--- |
| `catId` | INTEGER | PK (Part 1) | PFS カタログ識別番号（Catalogue ID） |
| `objId` | INTEGER | PK (Part 2) | PFS 天体識別番号（Object ID） |
| `combination` | INTEGER | PK (Part 3) | 露出結合インデックス（Combination Index） |
| `pred_class` | TEXT | NOT NULL | 予測天体クラス（`GALAXY`, `STAR`, `QSO`） |
| `confidence` | REAL | NOT NULL | 予測信頼度（最大予測確率, 0.0 〜 1.0） |
| `prob_galaxy` | REAL | NOT NULL | 銀河である予測確率（0.0 〜 1.0） |
| `prob_qso` | REAL | NOT NULL | クエーサーである予測確率（0.0 〜 1.0） |
| `prob_star` | REAL | NOT NULL | 恒星である予測確率（0.0 〜 1.0） |
| `est_redshift` | REAL | | 推定分光赤方偏移（GALAXY/QSO: 推定値, STAR: 0.0） |
| `est_redshift_quality` | REAL | | 赤方偏移推定品質（QSO: 相互相関ピーク値, GALAXY: 回帰信頼スコア） |

```sql
CREATE TABLE survey_predictions (
    catId INTEGER NOT NULL,
    objId INTEGER NOT NULL,
    combination INTEGER NOT NULL,
    pred_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    prob_galaxy REAL NOT NULL,
    prob_qso REAL NOT NULL,
    prob_star REAL NOT NULL,
    est_redshift REAL,
    est_redshift_quality REAL,
    PRIMARY KEY (catId, objId, combination)
);
```

### 7.4 カタログ生成とアクセスの Python コード例

#### カタログ生成スクリプトの実行:
```bash
python create_216k_full_survey_predictions_catalog.py
```

#### Python からの SQLite クエリ例:
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("results/pfs_full_survey_predictions.sqlite3")

# 高信頼度クエーサー (P(QSO) > 0.8) の抽出
query = """
SELECT catId, objId, combination, pred_class, prob_qso, est_redshift, est_redshift_quality
FROM survey_predictions
WHERE pred_class = 'QSO' AND prob_qso >= 0.8
ORDER BY prob_qso DESC
LIMIT 10;
"""
df_qso = pd.read_sql_query(query, conn)
print(df_qso)
conn.close()
```

---

## 8. 主要な診断結果の見方

1. **`latent_distribution_boost3x_smote.png`**:
   - **[A-1] True Label**: 真のラベルによる t-SNE 分布。QSO が孤立した明瞭なクラスタを形成。
   - **[A-2] Pred Label**: モデルの予測ラベル。SMOTE により境界部でも QSO が高精度に捕捉される様子。
   - **[B-1] P(QSO) Surface**: QSO 予測確率の連続グラデーション。
   - **[B-2] Agreement Map**: DB との一致・不一致天体の空間分布。
2. **`rf_disagreement_restframe_spectra_boost3x.png`**:
   - **New QSO Candidates (DB: GAL $\to$ Pred: QSO)**: 静止系スタックで強い Mg II, C IV が現れ、真の QSO であることを確認。
   - **Agreed QSO**: 典型的なブロードライン AGN スペクトル。
3. **`individual_missed_qsos_latent_positions.png`**:
   - 128 次元コサイン類似度空間（$\mathrm{Sim}_{\mathrm{QSO}}$ vs $\mathrm{Sim}_{\mathrm{GAL}}$）において、見逃された天体がなぜ銀河側に分類されたのかを幾何学的に立証。境界線上の天体（#5, $z=1.794$）は確率閾値の調整（$P(\text{QSO}) \ge 0.35$）により回収可能。
