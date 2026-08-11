# xyz-plot-split-by-batch

Stable Diffusion WebUI (Forge / Forge Neo) の標準スクリプトである「X/Y/Z Plot」を拡張し、バッチ生成時（Batch size > 1）に各バッチ単位で独立したグリッド画像を生成・保存できるようにするカスタムスクリプトです。

## 概要

通常の X/Y/Z Plot では、`Batch size` を 2 以上に設定すると全画像が 1 枚の巨大なグリッドにまとめられてしまい、1 枚ごとの画像が小さくなって確認しづらくなる問題がありました。

本スクリプト（**X/Y/Z plot split by batch**）を導入することで、バッチ内の各インデックス（1枚目、2枚目...）ごとに独立したグリッド画像が生成されます。同じプロンプトやパラメータ設定で、異なるシード値や構図による挙動の違いをバッチ単位で比較・検証したい場合に便利です。

## 主な機能・特徴

* **バッチ単位のグリッド分割**
  * `Batch size` を指定した際、バッチインデックスごとに個別グリッドを出力します。
* **軸ごとの凡例表示制御 (Draw legend for X/Y)**
  * X軸・Y軸の凡例（ラベル）表示を個別に ON/OFF 切り替え可能です。
* **改行区切り入力 (Use newline separator)**
  * 軸の入力欄をカンマ（`,`）区切りではなく、改行区切りで記述できます。
  * プロンプト比較の際、`""` や `''` を記述するか空行（改行2回）を作ることで、「プロンプト要素あり vs なし」の比較が簡単に設定できます。
* **16:9 アスペクト比への自動折り返し**
  * X軸のみ設定時の `Row Count = 0` (Auto) 設定において、余白や凡例を含めた最終出力画像全体が 16:9（ディスプレイ表示向け）に近くなるよう自動的に最適な段数へ折り返されます。

## インストール方法

1. WebUI の `Extensions` タブを開きます。
2. `Install from URL` タブを選択します。
3. `URL for extension's git repository` にこのリポジトリの URL を入力し、`Install` をクリックします。
4. インストール完了後、WebUI を再起動（Reload UI）します。

# xyz-plot-split-by-batch

An extended custom script for Stable Diffusion WebUI (Forge / Forge Neo) that splits X/Y/Z Plot grids by batch index when generating with a batch size greater than 1 (Batch size > 1).

## Overview

In the standard X/Y/Z Plot, setting Batch size to 2 or more combines all generated images into a single massive grid, making each individual image tiny and difficult to inspect.

This custom script (X/Y/Z plot split by batch) generates a separate, full-sized grid for each batch index (1st image, 2nd image, etc.). This makes it much easier to compare and verify variations in composition, seeds, or parameters across batches.

## Key Features
* **Batch-Based Grid Splitting**
  * When Batch size is set above 1, a separate comparison grid is saved for each batch image index.
* **Independent Legend Control (Draw legend for X/Y)**
  * Toggle legend labels independently for the X and Y axes.
* **Newline Separator Input (Use newline separator)**
  * Enter prompt values line-by-line instead of using commas (,).
  * Easily set up "with vs. without" comparisons (e.g., Prompt S/R) by entering "", '', or creating an empty line (pressing Enter twice).
* **Automatic 16:9 Aspect Ratio Layout**
  * When using only the X-axis with Row Count = 0 (Auto), the script calculates margins and legend heights to automatically wrap images into a layout close to a 16:9 aspect ratio.

## Installation
1. Open the Extensions tab in WebUI.
2. Select the Install from URL tab.
3. Paste this repository's URL into URL for extension's git repository and click Install.
4. After installation finishes, restart or reload the WebUI (Reload UI).
