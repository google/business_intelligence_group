# CausalImpact with Experimental Design 引き継ぎ資料

## 1. 目的と概要 (Overview)
このJupyter Notebook (`CausalImpact_with_Experimental_Design.ipynb`) は、マーケティング施策などの効果検証を行うための**因果推論 (Causal Impact Analysis)** と、その事前準備となる**実験計画 (Experimental Design)** を、ノーコード（UIウィジェット経由）でインタラクティブに実行するためのツールです。

ユーザーはGoogle Colab等の環境でこのNotebookを実行するだけで、UIからデータを読み込み、類似するコントロールグループの選定から、施策効果の推定までを一貫して行うことができます。

## 2. 前提条件と環境 (Prerequisites & Environment)
- **実行環境**: Google Colaboratory (推奨) または Jupyter Notebook環境。
- **主要な依存ライブラリ**:
  - `tfp-causalimpact` (TensorFlow ProbabilityベースのCausalImpact実装)
  - `tslearn` (時系列データのクラスタリング用。※Numbaエラー回避のため `==0.7.0` を指定)
  - `fastdtw` (Dynamic Time Warping距離計算用)
  - `altair` (インタラクティブなグラフ描画用)
  - `ipywidgets` (GUI構築用)
- **外部サービス連携**: Google Cloud (BigQuery), Google Sheetsへの認証 (`google.colab.auth`) を使用してデータをロードします。

## 3. 内部アーキテクチャ (Architecture & Classes)
Notebookは1つの巨大なコードセルにまとめられていますが、内部はオブジェクト指向およびSOLID原則に基づいてクラス分割されています。

- **`CausalImpactAnalysis` (Orchestrator)**
  - 全体の処理を統括するメインクラス。以下の各コンポーネントを保持し、処理を橋渡しします。
- **`InteractiveUI`**
  - `ipywidgets` を使ったUIの構築とイベントハンドリングを担当します。設定パラメータの保存(pickle)・読み込みも行います。
- **`DataLoader` (データ読み込み)**
  - `IDataLoader` インターフェース（Protocol）を利用したStrategyパターンを採用。
  - 実装クラス: `GoogleSheetLoader`, `CSVLoader`, `BigQueryLoader`。新しいデータソースを追加する場合は、このProtocolに従ってクラスを追加します（OCP原則）。
- **`DataPreprocessor` (データ前処理)**
  - 指定されたKPI列や日付列を用いて、分析に適した形（主にWide形式の時系列データ）にデータを整形します。
- **`ExploratoryDataAnalyzer` (探索的データ分析)**
  - データの品質チェック（欠損値等）や、DTWを用いた時系列のトレンド・クラスタリング可視化を行います。
- **`ExperimentalDesigner` (実験計画)**
  - 施策対象（ターゲット）の時系列の動きと最も類似する対照群（コントロール）の組み合わせを、MAPE（平均絶対パーセンテージ誤差）などの指標を用いて探索・最適化します。
- **`SimulationOptimizer` (シミュレーション)**
  - 施策を実施した場合の効果サイズ（Treat Impact）や期間を仮定し、因果推論が正しく検出できるかのシミュレーションを実行します。
- **`CausalImpactEstimator` (効果検証)**
  - `tfp-causalimpact` をラップし、実際の事前・事後期間のデータを元に因果推論モデルを学習し、結果をグラフ化します。

## 4. UIと操作手順 (UI Operations)
1. **Data Source (データソース)**:
   - スプレッドシートURL、CSVアップロード、またはBigQueryのプロジェクト/テーブル名を指定してデータを読み込みます。
2. **Data Format (データフォーマット)**:
   - 日付列 (`Date column`)、KPI列、およびピボットが必要な場合はその設定を行い、データを時系列形式に整形します。
3. **Purpose (目的の選択)**:
   - **`experimental_design`**: 過去のデータを用いて、ターゲットと連動して動くコントロール群の探索とシミュレーションを行います。
   - **`causal_impact`**: 実際に施策を実施した後のデータを用いて、その施策がどれだけのインパクトを与えたか（リフト効果）を推定します。

## 5. 運用・保守のポイント (Maintenance)
- **1ファイル構成の維持**:
  - ユーザーが手軽に実行できるよう、複数の `.py` ファイルに分割せず、1つの Notebookファイル (`.ipynb`) 内に全てのクラスを記述しています。
  - そのため、コードの修正や機能追加を行う場合は、Notebook内の該当クラス（例: `DataPreprocessor`）を直接編集します。
- **データソースの拡張**:
  - API連携など新しいデータロード元が必要になった場合は、`IDataLoader` を継承したクラスを新規作成し、`DataLoader` の条件分岐 (`source_index`) に追加するだけで拡張可能です。
- **スタイルガイド**:
  - 今後の保守においても、引数・戻り値の Type Hints と Google Styleの英語Docstring を記述するルールを維持してください。
