# exp0001
 * シンプルなベースライン
 * cutmix
 * マスク有りptach画像のみ使用
 * LB 0.922

# exp0002
 * exp0001にbatchの20%にはマスクなしの画像を追加するように変更
 * LB 0.923

# exp0003
 * 工夫
 * mixup系を色々と試した
 * cvがめちゃくちゃ上がりにくくなった、LBどうなるんだ？
 * LB 0.924

# exp0004
 * lossを少し変えてみてどうなるかを見てみたい
 * CV的にはsymmetric lovasz sigmoidが良かったけどLBはそんなに上がらんかった？

# exp0005
 * exp0003からの派生、hard DA
 * LB 0.922....

# exp0006
 * exp0002からの派生でtest画像だけで学習->exp0002からの派生でtrain画像でfinetune

