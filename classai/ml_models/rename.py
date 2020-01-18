import glob, os

# ファイル列挙
files = glob.glob("*.jpeg" or "*.jpg")

# 処理部分
for i, old_name in enumerate(files):
    # ファイル名を作成
    new_name = "img{0:03d}.jpg".format(i+1)
    # 変更実行
    os.rename(old_name, new_name)
    # 結果表示
    print(old_name + "→" + new_name)