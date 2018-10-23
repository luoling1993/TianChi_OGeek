# TianChi_OGeek

- 数据问题:train部分数据第1815102行漏了个引号，手动删除或不上引号即可

- 代码进行了更新，之前版本的代码通过tag回滚

- 当前版本代码实现了通过词向量作为文本特征进行文本聚类，需要耗费内存较大，低于16G请勿尝试

- 由于ignore了原始文件，项目文件树如下:

  >.
  >│  .gitignore
  >│  features_engineering.py
  >│  lgb_models.py
  >│  main.py
  >│  predict.csv
  >│  tree.txt
  >│  utils.py
  >│  w2v.bin
  >│  w2v.py
  >│  
  >│          
  >├─data
  >│  ├─EtlData
  >│  │      test.csv
  >│  │      train.csv
  >│  │      validate.csv
  >│  │      
  >│  └─RawData
  >│          oppo_round1_test_A.txt
  >│          oppo_round1_train.txt
  >│          oppo_round1_vali.txt
  >
  >

