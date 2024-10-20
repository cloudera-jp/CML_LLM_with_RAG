#!/bin/bash
# Pythonファイルの先頭に以下の3行を追加するためのスクリプト

LINE1="__import__('pysqlite3')"
LINE2="import sys"
LINE3="sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')"

# 対象の Python ファイル
FILE="/home/cdsw/.local/lib/python3.10/site-packages/chromadb/__init__.py"

# 一時保存用ファイル
TEMP_FILE=$(mktemp)

# ファイルの存在チェック
if [ ! -f "$FILE" ]; then
    echo "The specified file does not exist."
    exit 1
fi

# ロガーの処理を削除
sed -i '/logger = logging\.getLogger(__name__)/d' "$FILE"

# 該当の3行を一時ファイルに書き出し
echo "$LINE1" > "$TEMP_FILE"
echo "$LINE2" >> "$TEMP_FILE"
echo "$LINE3" >> "$TEMP_FILE"

# temp ファイルにもとのファイルを append 
cat "$FILE" >> "$TEMP_FILE"

# もとのファイルを temp の内容で上書き
mv "$TEMP_FILE" "$FILE"

echo "Lines added successfully."

# YAML ファイルのパスを定義
yaml_file="/home/cdsw/.local/lib/python3.10/site-packages/chromadb/log_config.yml"

# uvicorn 関係のログレベルが 'level: INFO' なら 'level: ERROR' に置き換え
sed -i '/uvicorn:/{n;s/level: INFO/level: ERROR/;}' "$yaml_file"