from helper import *

class InteractE(torch.nn.Module):
	"""
	Proposed method in the paper. Refer Section 6 of the paper for mode details 

	Parameters
	----------
	params:        	Hyperparameters of the model
	chequer_perm:   Reshaping to be used by the model
	
	Returns
	-------
	The InteractE model instance
		
	"""
	def __init__(self, params, chequer_perm):
		super(InteractE, self).__init__()
		#从终端输入的参数
		self.p                  = params
		#生成entity的embedding实例，使embedding的字典呈xavier_normal_分布
		self.ent_embed		= torch.nn.Embedding(self.p.num_ent,   self.p.embed_dim, padding_idx=None); xavier_normal_(self.ent_embed.weight)
		#生成relation的embedding实例，使embedding的字典呈xavier_normal_分布
		self.rel_embed		= torch.nn.Embedding(self.p.num_rel*2, self.p.embed_dim, padding_idx=None); xavier_normal_(self.rel_embed.weight)
		#定义损失函数
		self.bceloss		= torch.nn.BCELoss()

		#inp_drop方法以self.p.inp_drop的概率归零一些元素
		self.inp_drop		= torch.nn.Dropout(self.p.inp_drop)
		#hidden_drop方法以self.p.inp_drop的概率归零一些元素
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		#feature_map_drop方法以self.p.feat_drop的概率归零一些元素
		self.feature_map_drop	= torch.nn.Dropout2d(self.p.feat_drop)
		#Batch Normalization的目的是使我们的一批(Batch)Feature map满足均值等于0，方差等于1的分布规律
		self.bn0		= torch.nn.BatchNorm2d(self.p.perm)
		#Height of the reshaped matrix
		flat_sz_h 		= self.p.k_h
		#Width of the reshaped matrix
		flat_sz_w 		= 2*self.p.k_w
		#填充
		self.padding 		= 0
		#在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
		self.bn1 		= torch.nn.BatchNorm2d(self.p.num_filt*self.p.perm)
		#定义张量展平后的向量长度
		self.flat_sz 		= flat_sz_h * flat_sz_w * self.p.num_filt*self.p.perm
		#归一化处理
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		#定义一个全连接层，输入维度是flat_sz，输出维度是p.embed_dim
		self.fc 		= torch.nn.Linear(self.flat_sz, self.p.embed_dim)
		#定义重塑函数类型
		self.chequer_perm	= chequer_perm
		#Adds a parameter to the module.The parameter can be accessed as an attribute using given name.
		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
		#定义卷积核:(96,1,9,9).卷积channels=96
		self.register_parameter('conv_filt', Parameter(torch.zeros(self.p.num_filt, 1, self.p.ker_sz,  self.p.ker_sz))); xavier_normal_(self.conv_filt)
	#定义损失函数，一个常规的bceloss
	def loss(self, pred, true_label=None, sub_samp=None):
		label_pos	= true_label[0]; 
		label_neg	= true_label[1:]
		loss 		= self.bceloss(pred, true_label)
		return loss

	def circular_padding_chw(self, batch, padding):
		upper_pad	= batch[..., -padding:, :]
		lower_pad	= batch[..., :padding, :]
		#https://pytorch.org/docs/stable/generated/torch.cat.html?highlight=torch%20cat#torch.cat
		temp		= torch.cat([upper_pad, batch, lower_pad], dim=2)

		left_pad	= temp[..., -padding:]
		right_pad	= temp[..., :padding]
		padded		= torch.cat([left_pad, temp, right_pad], dim=3)
		return padded
	#神经网络流程
	def forward(self, sub, rel, neg_ents, strategy='one_to_x'):
		#实体嵌入,这里的sub_emb是列向量或者行向量
		sub_emb		= self.ent_embed(sub)
		#关系嵌入
		rel_emb		= self.rel_embed(rel)
		#torch.cat()按行或者按列合并张量.comb_emb是sub_emb和rel_emb按列合并的嵌入.
		comb_emb	= torch.cat([sub_emb, rel_emb], dim=1)
		#定义重塑之后的方格排列
		#x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据
		#在这里，chequer_perm还是一个一维tensor
		chequer_perm	= comb_emb[:, self.chequer_perm]
		#reshape在不改变数据总量的情况下改变原有张量的形状,"-1"维度会根据其它维度来计算，默认k_w=10,k_h=20.
		stack_inp	= chequer_perm.reshape((-1, self.p.perm, 2*self.p.k_w, self.p.k_h))
		stack_inp	= self.bn0(stack_inp)
		#第一层dropout
		x		= self.inp_drop(stack_inp)
		#环形填充,x.shape=11
		x		= self.circular_padding_chw(x, self.p.ker_sz//2)
		#第一层卷积，input=x, weight=(96,1,9,9)
		x		= F.conv2d(x, self.conv_filt.repeat(self.p.perm, 1, 1, 1), padding=self.padding, groups=self.p.perm)
		#batchnormal操作
		x		= self.bn1(x)
		#激活函数relu
		x		= F.relu(x)
		#dropout层
		x		= self.feature_map_drop(x)
		#更改tensor的形状，但是必须保持调整前后元素总和一致。此处为压平操作。
		x		= x.view(-1, self.flat_sz)
		#全连接层
		x		= self.fc(x)
		#dropout层
		x		= self.hidden_drop(x)
		#batchnormal操作
		x		= self.bn2(x)
		#激活函数relu
		x		= F.relu(x)
		#一对多时
		if strategy == 'one_to_n':
			x = torch.mm(x, self.ent_embed.weight.transpose(1,0))
			x += self.bias.expand_as(x)
		else:
			x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)
			x += self.bias[neg_ents]
		#预测值predict.使用sigmoid()选出可能性最大的值.
		pred	= torch.sigmoid(x)

		return pred