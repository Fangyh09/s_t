# Sequeece Tagging

目标: 88

## Features

- [ ] 使用Elmo
- [ ] 增加attention
- [ ] bie, bieo格式，其他格式
- [ ] reverse 文件
- [x] check文件
- [ ] 测试gru
- [ ] 训练模型，生成结果
    - [ ] 自动训练100轮非CNN版本`[adam, lr=1e-3, decay=0.9/0.95]`
    
          
          layers=2, clip=[0, 5];   
          layers=1, clip=[5]
          
    - [ ] 自动训练100轮CNN版本`[adam, lr=1e-3, decay=0.9/0.95]`
    
          filter_sizes=[3,4] lstm_layers=2
          filter_sizes=[3,4,5] lstm_layers=2
          
    - [ ] 手工训练非CNN版本每次延迟降低 `[adam, lr=1e-3, decay=0.9]`
    
          layers=2, clip=[5]
          
    - [ ] 手工训练CNN版本每次延迟降低`[adam, lr=1e-3, decay=0.9/0.95]`
    
          filter_sizes=[3,4] lstm_layers=2
          filter_sizes=[3,4,5] lstm_layers=2
          
        



