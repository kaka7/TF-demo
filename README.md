* 直接修改mnist数据位置，然后 python ./CNN/tf_run_CNN_visulization.py
        
        ./ckpt_dir/model.ckpt-2
        ('Start from:', 2)
        iter 0: the correct_rate is 0.9453125 
        iter 10: the correct_rate is 0.9765625 
        iter 20: the correct_rate is 0.9453125 
        iter 30: the correct_rate is 0.953125
        iter 40: the correct_rate is 0.9765625
        iter 50: the correct_rate is 0.96875
        iter 60: the correct_rate is 0.953125
        iter 70: the correct_rate is 0.9765625
        iter 80: the correct_rate is 0.9765625
        iter 90: the correct_rate is 0.953125
        Optimization Completed
        the test data sets' correct_rate is 0.953125
        
        checkpoint 
        可实现每次加载上次训练的结果
        
        该模板同时上课通过tensorboard查看训练过程