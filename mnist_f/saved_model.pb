??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv2d_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_57/kernel
}
$conv2d_57/kernel/Read/ReadVariableOpReadVariableOpconv2d_57/kernel*&
_output_shapes
: *
dtype0
t
conv2d_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_57/bias
m
"conv2d_57/bias/Read/ReadVariableOpReadVariableOpconv2d_57/bias*
_output_shapes
: *
dtype0
?
batch_normalization_43/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_43/gamma
?
0batch_normalization_43/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_43/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_43/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_43/beta
?
/batch_normalization_43/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_43/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_43/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_43/moving_mean
?
6batch_normalization_43/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_43/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_43/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_43/moving_variance
?
:batch_normalization_43/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_43/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_58/kernel
}
$conv2d_58/kernel/Read/ReadVariableOpReadVariableOpconv2d_58/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_58/bias
m
"conv2d_58/bias/Read/ReadVariableOpReadVariableOpconv2d_58/bias*
_output_shapes
: *
dtype0
?
batch_normalization_44/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_44/gamma
?
0batch_normalization_44/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_44/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_44/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_44/beta
?
/batch_normalization_44/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_44/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_44/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_44/moving_mean
?
6batch_normalization_44/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_44/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_44/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_44/moving_variance
?
:batch_normalization_44/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_44/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_59/kernel
}
$conv2d_59/kernel/Read/ReadVariableOpReadVariableOpconv2d_59/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_59/bias
m
"conv2d_59/bias/Read/ReadVariableOpReadVariableOpconv2d_59/bias*
_output_shapes
:@*
dtype0
?
conv2d_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_60/kernel
~
$conv2d_60/kernel/Read/ReadVariableOpReadVariableOpconv2d_60/kernel*'
_output_shapes
:@?*
dtype0
u
conv2d_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_60/bias
n
"conv2d_60/bias/Read/ReadVariableOpReadVariableOpconv2d_60/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_45/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_45/gamma
?
0batch_normalization_45/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_45/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_45/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_45/beta
?
/batch_normalization_45/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_45/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_45/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_45/moving_mean
?
6batch_normalization_45/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_45/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_45/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_45/moving_variance
?
:batch_normalization_45/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_45/moving_variance*
_output_shapes	
:?*
dtype0
|
dense_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?Q?* 
shared_namedense_36/kernel
u
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel* 
_output_shapes
:
?Q?*
dtype0
s
dense_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_36/bias
l
!dense_36/bias/Read/ReadVariableOpReadVariableOpdense_36/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_46/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_46/gamma
?
0batch_normalization_46/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_46/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_46/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_46/beta
?
/batch_normalization_46/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_46/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_46/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_46/moving_mean
?
6batch_normalization_46/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_46/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_46/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_46/moving_variance
?
:batch_normalization_46/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_46/moving_variance*
_output_shapes	
:?*
dtype0
{
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
* 
shared_namedense_37/kernel
t
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel*
_output_shapes
:	?
*
dtype0
r
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_37/bias
k
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes
:
*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
?T
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?T
value?TB?T B?T
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
	optimizer

signatures
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
?
axis
	 gamma
!beta
"moving_mean
#moving_variance
#$_self_saveable_object_factories
%regularization_losses
&trainable_variables
'	variables
(	keras_api
?

)kernel
*bias
#+_self_saveable_object_factories
,regularization_losses
-trainable_variables
.	variables
/	keras_api
?
0axis
	1gamma
2beta
3moving_mean
4moving_variance
#5_self_saveable_object_factories
6regularization_losses
7trainable_variables
8	variables
9	keras_api
w
#:_self_saveable_object_factories
;regularization_losses
<trainable_variables
=	variables
>	keras_api
?

?kernel
@bias
#A_self_saveable_object_factories
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
w
#F_self_saveable_object_factories
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
w
#K_self_saveable_object_factories
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
?

Pkernel
Qbias
#R_self_saveable_object_factories
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
?
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
#\_self_saveable_object_factories
]regularization_losses
^trainable_variables
_	variables
`	keras_api
w
#a_self_saveable_object_factories
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
w
#f_self_saveable_object_factories
gregularization_losses
htrainable_variables
i	variables
j	keras_api
?

kkernel
lbias
#m_self_saveable_object_factories
nregularization_losses
otrainable_variables
p	variables
q	keras_api
?
raxis
	sgamma
tbeta
umoving_mean
vmoving_variance
#w_self_saveable_object_factories
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
x
#|_self_saveable_object_factories
}regularization_losses
~trainable_variables
	variables
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
 
 
?
0
1
 2
!3
)4
*5
16
27
?8
@9
P10
Q11
X12
Y13
k14
l15
s16
t17
?18
?19
?
0
1
 2
!3
"4
#5
)6
*7
18
29
310
411
?12
@13
P14
Q15
X16
Y17
Z18
[19
k20
l21
s22
t23
u24
v25
?26
?27
?
?non_trainable_variables
?metrics
regularization_losses
trainable_variables
 ?layer_regularization_losses
?layers
	variables
?layer_metrics
\Z
VARIABLE_VALUEconv2d_57/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_57/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
?non_trainable_variables
?metrics
regularization_losses
trainable_variables
 ?layer_regularization_losses
?layers
	variables
?layer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_43/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_43/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_43/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_43/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

 0
!1

 0
!1
"2
#3
?
?non_trainable_variables
?metrics
%regularization_losses
&trainable_variables
 ?layer_regularization_losses
?layers
'	variables
?layer_metrics
\Z
VARIABLE_VALUEconv2d_58/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_58/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

)0
*1

)0
*1
?
?non_trainable_variables
?metrics
,regularization_losses
-trainable_variables
 ?layer_regularization_losses
?layers
.	variables
?layer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_44/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_44/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_44/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_44/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

10
21

10
21
32
43
?
?non_trainable_variables
?metrics
6regularization_losses
7trainable_variables
 ?layer_regularization_losses
?layers
8	variables
?layer_metrics
 
 
 
 
?
?non_trainable_variables
?metrics
;regularization_losses
<trainable_variables
 ?layer_regularization_losses
?layers
=	variables
?layer_metrics
\Z
VARIABLE_VALUEconv2d_59/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_59/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
@1

?0
@1
?
?non_trainable_variables
?metrics
Bregularization_losses
Ctrainable_variables
 ?layer_regularization_losses
?layers
D	variables
?layer_metrics
 
 
 
 
?
?non_trainable_variables
?metrics
Gregularization_losses
Htrainable_variables
 ?layer_regularization_losses
?layers
I	variables
?layer_metrics
 
 
 
 
?
?non_trainable_variables
?metrics
Lregularization_losses
Mtrainable_variables
 ?layer_regularization_losses
?layers
N	variables
?layer_metrics
\Z
VARIABLE_VALUEconv2d_60/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_60/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

P0
Q1

P0
Q1
?
?non_trainable_variables
?metrics
Sregularization_losses
Ttrainable_variables
 ?layer_regularization_losses
?layers
U	variables
?layer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_45/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_45/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_45/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_45/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

X0
Y1

X0
Y1
Z2
[3
?
?non_trainable_variables
?metrics
]regularization_losses
^trainable_variables
 ?layer_regularization_losses
?layers
_	variables
?layer_metrics
 
 
 
 
?
?non_trainable_variables
?metrics
bregularization_losses
ctrainable_variables
 ?layer_regularization_losses
?layers
d	variables
?layer_metrics
 
 
 
 
?
?non_trainable_variables
?metrics
gregularization_losses
htrainable_variables
 ?layer_regularization_losses
?layers
i	variables
?layer_metrics
[Y
VARIABLE_VALUEdense_36/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_36/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

k0
l1

k0
l1
?
?non_trainable_variables
?metrics
nregularization_losses
otrainable_variables
 ?layer_regularization_losses
?layers
p	variables
?layer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_46/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_46/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_46/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_46/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

s0
t1

s0
t1
u2
v3
?
?non_trainable_variables
?metrics
xregularization_losses
ytrainable_variables
 ?layer_regularization_losses
?layers
z	variables
?layer_metrics
 
 
 
 
?
?non_trainable_variables
?metrics
}regularization_losses
~trainable_variables
 ?layer_regularization_losses
?layers
	variables
?layer_metrics
[Y
VARIABLE_VALUEdense_37/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_37/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1

?0
?1
?
?non_trainable_variables
?metrics
?regularization_losses
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?layer_metrics
8
"0
#1
32
43
Z4
[5
u6
v7

?0
?1
 
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
 
 
 
 
 
 

"0
#1
 
 
 
 
 
 
 
 
 

30
41
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Z0
[1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

u0
v1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
?
serving_default_conv2d_57_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_57_inputconv2d_57/kernelconv2d_57/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv2d_58/kernelconv2d_58/biasbatch_normalization_44/gammabatch_normalization_44/beta"batch_normalization_44/moving_mean&batch_normalization_44/moving_varianceconv2d_59/kernelconv2d_59/biasconv2d_60/kernelconv2d_60/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_variancedense_36/kerneldense_36/bias&batch_normalization_46/moving_variancebatch_normalization_46/gamma"batch_normalization_46/moving_meanbatch_normalization_46/betadense_37/kerneldense_37/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_671673
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_57/kernel/Read/ReadVariableOp"conv2d_57/bias/Read/ReadVariableOp0batch_normalization_43/gamma/Read/ReadVariableOp/batch_normalization_43/beta/Read/ReadVariableOp6batch_normalization_43/moving_mean/Read/ReadVariableOp:batch_normalization_43/moving_variance/Read/ReadVariableOp$conv2d_58/kernel/Read/ReadVariableOp"conv2d_58/bias/Read/ReadVariableOp0batch_normalization_44/gamma/Read/ReadVariableOp/batch_normalization_44/beta/Read/ReadVariableOp6batch_normalization_44/moving_mean/Read/ReadVariableOp:batch_normalization_44/moving_variance/Read/ReadVariableOp$conv2d_59/kernel/Read/ReadVariableOp"conv2d_59/bias/Read/ReadVariableOp$conv2d_60/kernel/Read/ReadVariableOp"conv2d_60/bias/Read/ReadVariableOp0batch_normalization_45/gamma/Read/ReadVariableOp/batch_normalization_45/beta/Read/ReadVariableOp6batch_normalization_45/moving_mean/Read/ReadVariableOp:batch_normalization_45/moving_variance/Read/ReadVariableOp#dense_36/kernel/Read/ReadVariableOp!dense_36/bias/Read/ReadVariableOp0batch_normalization_46/gamma/Read/ReadVariableOp/batch_normalization_46/beta/Read/ReadVariableOp6batch_normalization_46/moving_mean/Read/ReadVariableOp:batch_normalization_46/moving_variance/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_672891
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_57/kernelconv2d_57/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv2d_58/kernelconv2d_58/biasbatch_normalization_44/gammabatch_normalization_44/beta"batch_normalization_44/moving_mean&batch_normalization_44/moving_varianceconv2d_59/kernelconv2d_59/biasconv2d_60/kernelconv2d_60/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_variancedense_36/kerneldense_36/biasbatch_normalization_46/gammabatch_normalization_46/beta"batch_normalization_46/moving_mean&batch_normalization_46/moving_variancedense_37/kerneldense_37/biastotalcounttotal_1count_1*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_672997??
?0
?
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_670659

inputs
assignmovingavg_670634
assignmovingavg_1_670640)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/670634*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_670634*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/670634*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/670634*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_670634AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/670634*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/670640*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_670640*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/670640*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/670640*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_670640AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/670640*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_57_layer_call_and_return_conditional_losses_672078

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_59_layer_call_and_return_conditional_losses_670924

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_670771

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
F__inference_dropout_62_layer_call_and_return_conditional_losses_671218

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?Q
?

I__inference_sequential_17_layer_call_and_return_conditional_losses_671551

inputs
conv2d_57_671478
conv2d_57_671480!
batch_normalization_43_671483!
batch_normalization_43_671485!
batch_normalization_43_671487!
batch_normalization_43_671489
conv2d_58_671492
conv2d_58_671494!
batch_normalization_44_671497!
batch_normalization_44_671499!
batch_normalization_44_671501!
batch_normalization_44_671503
conv2d_59_671507
conv2d_59_671509
conv2d_60_671514
conv2d_60_671516!
batch_normalization_45_671519!
batch_normalization_45_671521!
batch_normalization_45_671523!
batch_normalization_45_671525
dense_36_671530
dense_36_671532!
batch_normalization_46_671535!
batch_normalization_46_671537!
batch_normalization_46_671539!
batch_normalization_46_671541
dense_37_671545
dense_37_671547
identity??.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_46/StatefulPartitionedCall?!conv2d_57/StatefulPartitionedCall?!conv2d_58/StatefulPartitionedCall?!conv2d_59/StatefulPartitionedCall?!conv2d_60/StatefulPartitionedCall? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_57_671478conv2d_57_671480*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_57_layer_call_and_return_conditional_losses_6707182#
!conv2d_57/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0batch_normalization_43_671483batch_normalization_43_671485batch_normalization_43_671487batch_normalization_43_671489*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_67077120
.batch_normalization_43/StatefulPartitionedCall?
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0conv2d_58_671492conv2d_58_671494*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_58_layer_call_and_return_conditional_losses_6708182#
!conv2d_58/StatefulPartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_44_671497batch_normalization_44_671499batch_normalization_44_671501batch_normalization_44_671503*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_67087120
.batch_normalization_44/StatefulPartitionedCall?
dropout_59/PartitionedCallPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_59_layer_call_and_return_conditional_losses_6709242
dropout_59/PartitionedCall?
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall#dropout_59/PartitionedCall:output:0conv2d_59_671507conv2d_59_671509*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_59_layer_call_and_return_conditional_losses_6709482#
!conv2d_59/StatefulPartitionedCall?
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_6704532"
 max_pooling2d_23/PartitionedCall?
dropout_60/PartitionedCallPartitionedCall)max_pooling2d_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_60_layer_call_and_return_conditional_losses_6709822
dropout_60/PartitionedCall?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall#dropout_60/PartitionedCall:output:0conv2d_60_671514conv2d_60_671516*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_60_layer_call_and_return_conditional_losses_6710062#
!conv2d_60/StatefulPartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0batch_normalization_45_671519batch_normalization_45_671521batch_normalization_45_671523batch_normalization_45_671525*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_67105920
.batch_normalization_45/StatefulPartitionedCall?
dropout_61/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_61_layer_call_and_return_conditional_losses_6711122
dropout_61/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall#dropout_61/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_6711312
flatten_15/PartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_36_671530dense_36_671532*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_6711502"
 dense_36/StatefulPartitionedCall?
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0batch_normalization_46_671535batch_normalization_46_671537batch_normalization_46_671539batch_normalization_46_671541*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_67069220
.batch_normalization_46/StatefulPartitionedCall?
dropout_62/PartitionedCallPartitionedCall7batch_normalization_46/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_62_layer_call_and_return_conditional_losses_6712182
dropout_62/PartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_62/PartitionedCall:output:0dense_37_671545dense_37_671547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_6712422"
 dense_37/StatefulPartitionedCall?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_46/StatefulPartitionedCall"^conv2d_57/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_670692

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_17_layer_call_fn_671610
conv2d_57_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_57_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_6715512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_57_input
?
?
7__inference_batch_normalization_43_layer_call_fn_672215

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_6703322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
E__inference_conv2d_60_layer_call_and_return_conditional_losses_671006

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_670405

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

*__inference_conv2d_60_layer_call_fn_672457

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_60_layer_call_and_return_conditional_losses_6710062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
F__inference_flatten_15_layer_call_and_return_conditional_losses_672618

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????(  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Q2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????Q2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_672319

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_670853

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_672541

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_44_layer_call_fn_672299

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_6704362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_23_layer_call_fn_670459

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_6704532
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_60_layer_call_and_return_conditional_losses_672427

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

*__inference_conv2d_57_layer_call_fn_672087

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_57_layer_call_and_return_conditional_losses_6707182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
ך
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_671945

inputs,
(conv2d_57_conv2d_readvariableop_resource-
)conv2d_57_biasadd_readvariableop_resource2
.batch_normalization_43_readvariableop_resource4
0batch_normalization_43_readvariableop_1_resourceC
?batch_normalization_43_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_58_conv2d_readvariableop_resource-
)conv2d_58_biasadd_readvariableop_resource2
.batch_normalization_44_readvariableop_resource4
0batch_normalization_44_readvariableop_1_resourceC
?batch_normalization_44_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_59_conv2d_readvariableop_resource-
)conv2d_59_biasadd_readvariableop_resource,
(conv2d_60_conv2d_readvariableop_resource-
)conv2d_60_biasadd_readvariableop_resource2
.batch_normalization_45_readvariableop_resource4
0batch_normalization_45_readvariableop_1_resourceC
?batch_normalization_45_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource<
8batch_normalization_46_batchnorm_readvariableop_resource@
<batch_normalization_46_batchnorm_mul_readvariableop_resource>
:batch_normalization_46_batchnorm_readvariableop_1_resource>
:batch_normalization_46_batchnorm_readvariableop_2_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity??6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_43/ReadVariableOp?'batch_normalization_43/ReadVariableOp_1?6batch_normalization_44/FusedBatchNormV3/ReadVariableOp?8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_44/ReadVariableOp?'batch_normalization_44/ReadVariableOp_1?6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_45/ReadVariableOp?'batch_normalization_45/ReadVariableOp_1?/batch_normalization_46/batchnorm/ReadVariableOp?1batch_normalization_46/batchnorm/ReadVariableOp_1?1batch_normalization_46/batchnorm/ReadVariableOp_2?3batch_normalization_46/batchnorm/mul/ReadVariableOp? conv2d_57/BiasAdd/ReadVariableOp?conv2d_57/Conv2D/ReadVariableOp? conv2d_58/BiasAdd/ReadVariableOp?conv2d_58/Conv2D/ReadVariableOp? conv2d_59/BiasAdd/ReadVariableOp?conv2d_59/Conv2D/ReadVariableOp? conv2d_60/BiasAdd/ReadVariableOp?conv2d_60/Conv2D/ReadVariableOp?dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOp?
conv2d_57/Conv2D/ReadVariableOpReadVariableOp(conv2d_57_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_57/Conv2D/ReadVariableOp?
conv2d_57/Conv2DConv2Dinputs'conv2d_57/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_57/Conv2D?
 conv2d_57/BiasAdd/ReadVariableOpReadVariableOp)conv2d_57_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_57/BiasAdd/ReadVariableOp?
conv2d_57/BiasAddBiasAddconv2d_57/Conv2D:output:0(conv2d_57/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_57/BiasAdd~
conv2d_57/ReluReluconv2d_57/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_57/Relu?
%batch_normalization_43/ReadVariableOpReadVariableOp.batch_normalization_43_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_43/ReadVariableOp?
'batch_normalization_43/ReadVariableOp_1ReadVariableOp0batch_normalization_43_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_43/ReadVariableOp_1?
6batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_43/FusedBatchNormV3FusedBatchNormV3conv2d_57/Relu:activations:0-batch_normalization_43/ReadVariableOp:value:0/batch_normalization_43/ReadVariableOp_1:value:0>batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_43/FusedBatchNormV3?
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_58/Conv2D/ReadVariableOp?
conv2d_58/Conv2DConv2D+batch_normalization_43/FusedBatchNormV3:y:0'conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_58/Conv2D?
 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_58/BiasAdd/ReadVariableOp?
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0(conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_58/BiasAdd~
conv2d_58/ReluReluconv2d_58/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_58/Relu?
%batch_normalization_44/ReadVariableOpReadVariableOp.batch_normalization_44_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_44/ReadVariableOp?
'batch_normalization_44/ReadVariableOp_1ReadVariableOp0batch_normalization_44_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_44/ReadVariableOp_1?
6batch_normalization_44/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_44_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_44/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_44/FusedBatchNormV3FusedBatchNormV3conv2d_58/Relu:activations:0-batch_normalization_44/ReadVariableOp:value:0/batch_normalization_44/ReadVariableOp_1:value:0>batch_normalization_44/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_44/FusedBatchNormV3?
dropout_59/IdentityIdentity+batch_normalization_44/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2
dropout_59/Identity?
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_59/Conv2D/ReadVariableOp?
conv2d_59/Conv2DConv2Ddropout_59/Identity:output:0'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_59/Conv2D?
 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_59/BiasAdd/ReadVariableOp?
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_59/BiasAdd~
conv2d_59/ReluReluconv2d_59/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_59/Relu?
max_pooling2d_23/MaxPoolMaxPoolconv2d_59/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPool?
dropout_60/IdentityIdentity!max_pooling2d_23/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout_60/Identity?
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_60/Conv2D/ReadVariableOp?
conv2d_60/Conv2DConv2Ddropout_60/Identity:output:0'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv2d_60/Conv2D?
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_60/BiasAdd/ReadVariableOp?
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv2d_60/BiasAdd
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
conv2d_60/Relu?
%batch_normalization_45/ReadVariableOpReadVariableOp.batch_normalization_45_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_45/ReadVariableOp?
'batch_normalization_45/ReadVariableOp_1ReadVariableOp0batch_normalization_45_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_45/ReadVariableOp_1?
6batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_45/FusedBatchNormV3FusedBatchNormV3conv2d_60/Relu:activations:0-batch_normalization_45/ReadVariableOp:value:0/batch_normalization_45/ReadVariableOp_1:value:0>batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_45/FusedBatchNormV3?
dropout_61/IdentityIdentity+batch_normalization_45/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????		?2
dropout_61/Identityu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????(  2
flatten_15/Const?
flatten_15/ReshapeReshapedropout_61/Identity:output:0flatten_15/Const:output:0*
T0*(
_output_shapes
:??????????Q2
flatten_15/Reshape?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
?Q?*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMulflatten_15/Reshape:output:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/BiasAddt
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_36/Relu?
/batch_normalization_46/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_46_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_46/batchnorm/ReadVariableOp?
&batch_normalization_46/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_46/batchnorm/add/y?
$batch_normalization_46/batchnorm/addAddV27batch_normalization_46/batchnorm/ReadVariableOp:value:0/batch_normalization_46/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_46/batchnorm/add?
&batch_normalization_46/batchnorm/RsqrtRsqrt(batch_normalization_46/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_46/batchnorm/Rsqrt?
3batch_normalization_46/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_46_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_46/batchnorm/mul/ReadVariableOp?
$batch_normalization_46/batchnorm/mulMul*batch_normalization_46/batchnorm/Rsqrt:y:0;batch_normalization_46/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_46/batchnorm/mul?
&batch_normalization_46/batchnorm/mul_1Muldense_36/Relu:activations:0(batch_normalization_46/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_46/batchnorm/mul_1?
1batch_normalization_46/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_46_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_46/batchnorm/ReadVariableOp_1?
&batch_normalization_46/batchnorm/mul_2Mul9batch_normalization_46/batchnorm/ReadVariableOp_1:value:0(batch_normalization_46/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_46/batchnorm/mul_2?
1batch_normalization_46/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_46_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_46/batchnorm/ReadVariableOp_2?
$batch_normalization_46/batchnorm/subSub9batch_normalization_46/batchnorm/ReadVariableOp_2:value:0*batch_normalization_46/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_46/batchnorm/sub?
&batch_normalization_46/batchnorm/add_1AddV2*batch_normalization_46/batchnorm/mul_1:z:0(batch_normalization_46/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_46/batchnorm/add_1?
dropout_62/IdentityIdentity*batch_normalization_46/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????2
dropout_62/Identity?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldropout_62/Identity:output:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_37/BiasAdd|
dense_37/SoftmaxSoftmaxdense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_37/Softmax?

IdentityIdentitydense_37/Softmax:softmax:07^batch_normalization_43/FusedBatchNormV3/ReadVariableOp9^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_43/ReadVariableOp(^batch_normalization_43/ReadVariableOp_17^batch_normalization_44/FusedBatchNormV3/ReadVariableOp9^batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_44/ReadVariableOp(^batch_normalization_44/ReadVariableOp_17^batch_normalization_45/FusedBatchNormV3/ReadVariableOp9^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_45/ReadVariableOp(^batch_normalization_45/ReadVariableOp_10^batch_normalization_46/batchnorm/ReadVariableOp2^batch_normalization_46/batchnorm/ReadVariableOp_12^batch_normalization_46/batchnorm/ReadVariableOp_24^batch_normalization_46/batchnorm/mul/ReadVariableOp!^conv2d_57/BiasAdd/ReadVariableOp ^conv2d_57/Conv2D/ReadVariableOp!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::2p
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp6batch_normalization_43/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_18batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_43/ReadVariableOp%batch_normalization_43/ReadVariableOp2R
'batch_normalization_43/ReadVariableOp_1'batch_normalization_43/ReadVariableOp_12p
6batch_normalization_44/FusedBatchNormV3/ReadVariableOp6batch_normalization_44/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_18batch_normalization_44/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_44/ReadVariableOp%batch_normalization_44/ReadVariableOp2R
'batch_normalization_44/ReadVariableOp_1'batch_normalization_44/ReadVariableOp_12p
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp6batch_normalization_45/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_18batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_45/ReadVariableOp%batch_normalization_45/ReadVariableOp2R
'batch_normalization_45/ReadVariableOp_1'batch_normalization_45/ReadVariableOp_12b
/batch_normalization_46/batchnorm/ReadVariableOp/batch_normalization_46/batchnorm/ReadVariableOp2f
1batch_normalization_46/batchnorm/ReadVariableOp_11batch_normalization_46/batchnorm/ReadVariableOp_12f
1batch_normalization_46/batchnorm/ReadVariableOp_21batch_normalization_46/batchnorm/ReadVariableOp_22j
3batch_normalization_46/batchnorm/mul/ReadVariableOp3batch_normalization_46/batchnorm/mul/ReadVariableOp2D
 conv2d_57/BiasAdd/ReadVariableOp conv2d_57/BiasAdd/ReadVariableOp2B
conv2d_57/Conv2D/ReadVariableOpconv2d_57/Conv2D/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
+__inference_dropout_61_layer_call_fn_672607

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_61_layer_call_and_return_conditional_losses_6711072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
G
+__inference_dropout_62_layer_call_fn_672752

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_62_layer_call_and_return_conditional_losses_6712182
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_37_layer_call_and_return_conditional_losses_671242

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_670871

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_672125

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_672171

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_670753

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
.__inference_sequential_17_layer_call_fn_672006

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_6714142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_670332

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
d
+__inference_dropout_62_layer_call_fn_672747

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_62_layer_call_and_return_conditional_losses_6712132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_15_layer_call_and_return_conditional_losses_671131

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????(  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Q2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????Q2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_44_layer_call_fn_672350

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_6708532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_43_layer_call_fn_672151

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_6707712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_672997
file_prefix%
!assignvariableop_conv2d_57_kernel%
!assignvariableop_1_conv2d_57_bias3
/assignvariableop_2_batch_normalization_43_gamma2
.assignvariableop_3_batch_normalization_43_beta9
5assignvariableop_4_batch_normalization_43_moving_mean=
9assignvariableop_5_batch_normalization_43_moving_variance'
#assignvariableop_6_conv2d_58_kernel%
!assignvariableop_7_conv2d_58_bias3
/assignvariableop_8_batch_normalization_44_gamma2
.assignvariableop_9_batch_normalization_44_beta:
6assignvariableop_10_batch_normalization_44_moving_mean>
:assignvariableop_11_batch_normalization_44_moving_variance(
$assignvariableop_12_conv2d_59_kernel&
"assignvariableop_13_conv2d_59_bias(
$assignvariableop_14_conv2d_60_kernel&
"assignvariableop_15_conv2d_60_bias4
0assignvariableop_16_batch_normalization_45_gamma3
/assignvariableop_17_batch_normalization_45_beta:
6assignvariableop_18_batch_normalization_45_moving_mean>
:assignvariableop_19_batch_normalization_45_moving_variance'
#assignvariableop_20_dense_36_kernel%
!assignvariableop_21_dense_36_bias4
0assignvariableop_22_batch_normalization_46_gamma3
/assignvariableop_23_batch_normalization_46_beta:
6assignvariableop_24_batch_normalization_46_moving_mean>
:assignvariableop_25_batch_normalization_46_moving_variance'
#assignvariableop_26_dense_37_kernel%
!assignvariableop_27_dense_37_bias
assignvariableop_28_total
assignvariableop_29_count
assignvariableop_30_total_1
assignvariableop_31_count_1
identity_33??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::*/
dtypes%
#2!2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_57_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_57_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_43_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_43_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_43_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_43_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_58_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_58_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_44_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_44_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_44_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_44_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_59_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_59_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_60_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_60_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_45_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_45_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_45_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_45_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_36_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_36_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_batch_normalization_46_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batch_normalization_46_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp6assignvariableop_24_batch_normalization_46_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp:assignvariableop_25_batch_normalization_46_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp#assignvariableop_26_dense_37_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp!assignvariableop_27_dense_37_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_total_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_count_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_319
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_32?
Identity_33IdentityIdentity_32:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_33"#
identity_33Identity_33:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
e
F__inference_dropout_59_layer_call_and_return_conditional_losses_672375

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
E__inference_conv2d_57_layer_call_and_return_conditional_losses_670718

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_46_layer_call_fn_672712

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_6706592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_672189

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_671059

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????		?::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
d
F__inference_dropout_60_layer_call_and_return_conditional_losses_670982

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_670301

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
~
)__inference_dense_36_layer_call_fn_672643

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_6711502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????Q::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????Q
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_672337

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
e
F__inference_dropout_61_layer_call_and_return_conditional_losses_671107

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????		?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????		?*
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????		?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????		?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????		?2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_672273

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?W
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_671414

inputs
conv2d_57_671341
conv2d_57_671343!
batch_normalization_43_671346!
batch_normalization_43_671348!
batch_normalization_43_671350!
batch_normalization_43_671352
conv2d_58_671355
conv2d_58_671357!
batch_normalization_44_671360!
batch_normalization_44_671362!
batch_normalization_44_671364!
batch_normalization_44_671366
conv2d_59_671370
conv2d_59_671372
conv2d_60_671377
conv2d_60_671379!
batch_normalization_45_671382!
batch_normalization_45_671384!
batch_normalization_45_671386!
batch_normalization_45_671388
dense_36_671393
dense_36_671395!
batch_normalization_46_671398!
batch_normalization_46_671400!
batch_normalization_46_671402!
batch_normalization_46_671404
dense_37_671408
dense_37_671410
identity??.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_46/StatefulPartitionedCall?!conv2d_57/StatefulPartitionedCall?!conv2d_58/StatefulPartitionedCall?!conv2d_59/StatefulPartitionedCall?!conv2d_60/StatefulPartitionedCall? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?"dropout_59/StatefulPartitionedCall?"dropout_60/StatefulPartitionedCall?"dropout_61/StatefulPartitionedCall?"dropout_62/StatefulPartitionedCall?
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_57_671341conv2d_57_671343*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_57_layer_call_and_return_conditional_losses_6707182#
!conv2d_57/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0batch_normalization_43_671346batch_normalization_43_671348batch_normalization_43_671350batch_normalization_43_671352*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_67075320
.batch_normalization_43/StatefulPartitionedCall?
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0conv2d_58_671355conv2d_58_671357*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_58_layer_call_and_return_conditional_losses_6708182#
!conv2d_58/StatefulPartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_44_671360batch_normalization_44_671362batch_normalization_44_671364batch_normalization_44_671366*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_67085320
.batch_normalization_44/StatefulPartitionedCall?
"dropout_59/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_59_layer_call_and_return_conditional_losses_6709192$
"dropout_59/StatefulPartitionedCall?
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall+dropout_59/StatefulPartitionedCall:output:0conv2d_59_671370conv2d_59_671372*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_59_layer_call_and_return_conditional_losses_6709482#
!conv2d_59/StatefulPartitionedCall?
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_6704532"
 max_pooling2d_23/PartitionedCall?
"dropout_60/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_23/PartitionedCall:output:0#^dropout_59/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_60_layer_call_and_return_conditional_losses_6709772$
"dropout_60/StatefulPartitionedCall?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall+dropout_60/StatefulPartitionedCall:output:0conv2d_60_671377conv2d_60_671379*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_60_layer_call_and_return_conditional_losses_6710062#
!conv2d_60/StatefulPartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0batch_normalization_45_671382batch_normalization_45_671384batch_normalization_45_671386batch_normalization_45_671388*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_67104120
.batch_normalization_45/StatefulPartitionedCall?
"dropout_61/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0#^dropout_60/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_61_layer_call_and_return_conditional_losses_6711072$
"dropout_61/StatefulPartitionedCall?
flatten_15/PartitionedCallPartitionedCall+dropout_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_6711312
flatten_15/PartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_36_671393dense_36_671395*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_6711502"
 dense_36/StatefulPartitionedCall?
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0batch_normalization_46_671398batch_normalization_46_671400batch_normalization_46_671402batch_normalization_46_671404*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_67065920
.batch_normalization_46/StatefulPartitionedCall?
"dropout_62/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_46/StatefulPartitionedCall:output:0#^dropout_61/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_62_layer_call_and_return_conditional_losses_6712132$
"dropout_62/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall+dropout_62/StatefulPartitionedCall:output:0dense_37_671408dense_37_671410*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_6712422"
 dense_37/StatefulPartitionedCall?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_46/StatefulPartitionedCall"^conv2d_57/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall#^dropout_59/StatefulPartitionedCall#^dropout_60/StatefulPartitionedCall#^dropout_61/StatefulPartitionedCall#^dropout_62/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2H
"dropout_59/StatefulPartitionedCall"dropout_59/StatefulPartitionedCall2H
"dropout_60/StatefulPartitionedCall"dropout_60/StatefulPartitionedCall2H
"dropout_61/StatefulPartitionedCall"dropout_61/StatefulPartitionedCall2H
"dropout_62/StatefulPartitionedCall"dropout_62/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_45_layer_call_fn_672585

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_6705522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_flatten_15_layer_call_fn_672623

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_6711312
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????Q2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_670453

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_671673
conv2d_57_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_57_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_6702392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_57_input
??
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_671834

inputs,
(conv2d_57_conv2d_readvariableop_resource-
)conv2d_57_biasadd_readvariableop_resource2
.batch_normalization_43_readvariableop_resource4
0batch_normalization_43_readvariableop_1_resourceC
?batch_normalization_43_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_58_conv2d_readvariableop_resource-
)conv2d_58_biasadd_readvariableop_resource2
.batch_normalization_44_readvariableop_resource4
0batch_normalization_44_readvariableop_1_resourceC
?batch_normalization_44_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_59_conv2d_readvariableop_resource-
)conv2d_59_biasadd_readvariableop_resource,
(conv2d_60_conv2d_readvariableop_resource-
)conv2d_60_biasadd_readvariableop_resource2
.batch_normalization_45_readvariableop_resource4
0batch_normalization_45_readvariableop_1_resourceC
?batch_normalization_45_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource1
-batch_normalization_46_assignmovingavg_6717943
/batch_normalization_46_assignmovingavg_1_671800@
<batch_normalization_46_batchnorm_mul_readvariableop_resource<
8batch_normalization_46_batchnorm_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity??%batch_normalization_43/AssignNewValue?'batch_normalization_43/AssignNewValue_1?6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_43/ReadVariableOp?'batch_normalization_43/ReadVariableOp_1?%batch_normalization_44/AssignNewValue?'batch_normalization_44/AssignNewValue_1?6batch_normalization_44/FusedBatchNormV3/ReadVariableOp?8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_44/ReadVariableOp?'batch_normalization_44/ReadVariableOp_1?%batch_normalization_45/AssignNewValue?'batch_normalization_45/AssignNewValue_1?6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_45/ReadVariableOp?'batch_normalization_45/ReadVariableOp_1?:batch_normalization_46/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_46/AssignMovingAvg/ReadVariableOp?<batch_normalization_46/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_46/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_46/batchnorm/ReadVariableOp?3batch_normalization_46/batchnorm/mul/ReadVariableOp? conv2d_57/BiasAdd/ReadVariableOp?conv2d_57/Conv2D/ReadVariableOp? conv2d_58/BiasAdd/ReadVariableOp?conv2d_58/Conv2D/ReadVariableOp? conv2d_59/BiasAdd/ReadVariableOp?conv2d_59/Conv2D/ReadVariableOp? conv2d_60/BiasAdd/ReadVariableOp?conv2d_60/Conv2D/ReadVariableOp?dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOp?
conv2d_57/Conv2D/ReadVariableOpReadVariableOp(conv2d_57_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_57/Conv2D/ReadVariableOp?
conv2d_57/Conv2DConv2Dinputs'conv2d_57/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_57/Conv2D?
 conv2d_57/BiasAdd/ReadVariableOpReadVariableOp)conv2d_57_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_57/BiasAdd/ReadVariableOp?
conv2d_57/BiasAddBiasAddconv2d_57/Conv2D:output:0(conv2d_57/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_57/BiasAdd~
conv2d_57/ReluReluconv2d_57/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_57/Relu?
%batch_normalization_43/ReadVariableOpReadVariableOp.batch_normalization_43_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_43/ReadVariableOp?
'batch_normalization_43/ReadVariableOp_1ReadVariableOp0batch_normalization_43_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_43/ReadVariableOp_1?
6batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_43/FusedBatchNormV3FusedBatchNormV3conv2d_57/Relu:activations:0-batch_normalization_43/ReadVariableOp:value:0/batch_normalization_43/ReadVariableOp_1:value:0>batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_43/FusedBatchNormV3?
%batch_normalization_43/AssignNewValueAssignVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource4batch_normalization_43/FusedBatchNormV3:batch_mean:07^batch_normalization_43/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_43/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_43/AssignNewValue?
'batch_normalization_43/AssignNewValue_1AssignVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_43/FusedBatchNormV3:batch_variance:09^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_43/AssignNewValue_1?
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_58/Conv2D/ReadVariableOp?
conv2d_58/Conv2DConv2D+batch_normalization_43/FusedBatchNormV3:y:0'conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_58/Conv2D?
 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_58/BiasAdd/ReadVariableOp?
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0(conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_58/BiasAdd~
conv2d_58/ReluReluconv2d_58/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_58/Relu?
%batch_normalization_44/ReadVariableOpReadVariableOp.batch_normalization_44_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_44/ReadVariableOp?
'batch_normalization_44/ReadVariableOp_1ReadVariableOp0batch_normalization_44_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_44/ReadVariableOp_1?
6batch_normalization_44/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_44_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_44/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_44/FusedBatchNormV3FusedBatchNormV3conv2d_58/Relu:activations:0-batch_normalization_44/ReadVariableOp:value:0/batch_normalization_44/ReadVariableOp_1:value:0>batch_normalization_44/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_44/FusedBatchNormV3?
%batch_normalization_44/AssignNewValueAssignVariableOp?batch_normalization_44_fusedbatchnormv3_readvariableop_resource4batch_normalization_44/FusedBatchNormV3:batch_mean:07^batch_normalization_44/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_44/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_44/AssignNewValue?
'batch_normalization_44/AssignNewValue_1AssignVariableOpAbatch_normalization_44_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_44/FusedBatchNormV3:batch_variance:09^batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_44/AssignNewValue_1y
dropout_59/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_59/dropout/Const?
dropout_59/dropout/MulMul+batch_normalization_44/FusedBatchNormV3:y:0!dropout_59/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout_59/dropout/Mul?
dropout_59/dropout/ShapeShape+batch_normalization_44/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_59/dropout/Shape?
/dropout_59/dropout/random_uniform/RandomUniformRandomUniform!dropout_59/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0*
seed?21
/dropout_59/dropout/random_uniform/RandomUniform?
!dropout_59/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_59/dropout/GreaterEqual/y?
dropout_59/dropout/GreaterEqualGreaterEqual8dropout_59/dropout/random_uniform/RandomUniform:output:0*dropout_59/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2!
dropout_59/dropout/GreaterEqual?
dropout_59/dropout/CastCast#dropout_59/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout_59/dropout/Cast?
dropout_59/dropout/Mul_1Muldropout_59/dropout/Mul:z:0dropout_59/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout_59/dropout/Mul_1?
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_59/Conv2D/ReadVariableOp?
conv2d_59/Conv2DConv2Ddropout_59/dropout/Mul_1:z:0'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_59/Conv2D?
 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_59/BiasAdd/ReadVariableOp?
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_59/BiasAdd~
conv2d_59/ReluReluconv2d_59/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_59/Relu?
max_pooling2d_23/MaxPoolMaxPoolconv2d_59/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPooly
dropout_60/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_60/dropout/Const?
dropout_60/dropout/MulMul!max_pooling2d_23/MaxPool:output:0!dropout_60/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout_60/dropout/Mul?
dropout_60/dropout/ShapeShape!max_pooling2d_23/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_60/dropout/Shape?
/dropout_60/dropout/random_uniform/RandomUniformRandomUniform!dropout_60/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0*
seed?*
seed221
/dropout_60/dropout/random_uniform/RandomUniform?
!dropout_60/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_60/dropout/GreaterEqual/y?
dropout_60/dropout/GreaterEqualGreaterEqual8dropout_60/dropout/random_uniform/RandomUniform:output:0*dropout_60/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2!
dropout_60/dropout/GreaterEqual?
dropout_60/dropout/CastCast#dropout_60/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_60/dropout/Cast?
dropout_60/dropout/Mul_1Muldropout_60/dropout/Mul:z:0dropout_60/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_60/dropout/Mul_1?
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_60/Conv2D/ReadVariableOp?
conv2d_60/Conv2DConv2Ddropout_60/dropout/Mul_1:z:0'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv2d_60/Conv2D?
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_60/BiasAdd/ReadVariableOp?
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv2d_60/BiasAdd
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
conv2d_60/Relu?
%batch_normalization_45/ReadVariableOpReadVariableOp.batch_normalization_45_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_45/ReadVariableOp?
'batch_normalization_45/ReadVariableOp_1ReadVariableOp0batch_normalization_45_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_45/ReadVariableOp_1?
6batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_45/FusedBatchNormV3FusedBatchNormV3conv2d_60/Relu:activations:0-batch_normalization_45/ReadVariableOp:value:0/batch_normalization_45/ReadVariableOp_1:value:0>batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_45/FusedBatchNormV3?
%batch_normalization_45/AssignNewValueAssignVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource4batch_normalization_45/FusedBatchNormV3:batch_mean:07^batch_normalization_45/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_45/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_45/AssignNewValue?
'batch_normalization_45/AssignNewValue_1AssignVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_45/FusedBatchNormV3:batch_variance:09^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_45/AssignNewValue_1y
dropout_61/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_61/dropout/Const?
dropout_61/dropout/MulMul+batch_normalization_45/FusedBatchNormV3:y:0!dropout_61/dropout/Const:output:0*
T0*0
_output_shapes
:?????????		?2
dropout_61/dropout/Mul?
dropout_61/dropout/ShapeShape+batch_normalization_45/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_61/dropout/Shape?
/dropout_61/dropout/random_uniform/RandomUniformRandomUniform!dropout_61/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????		?*
dtype0*
seed?*
seed221
/dropout_61/dropout/random_uniform/RandomUniform?
!dropout_61/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_61/dropout/GreaterEqual/y?
dropout_61/dropout/GreaterEqualGreaterEqual8dropout_61/dropout/random_uniform/RandomUniform:output:0*dropout_61/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????		?2!
dropout_61/dropout/GreaterEqual?
dropout_61/dropout/CastCast#dropout_61/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????		?2
dropout_61/dropout/Cast?
dropout_61/dropout/Mul_1Muldropout_61/dropout/Mul:z:0dropout_61/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????		?2
dropout_61/dropout/Mul_1u
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????(  2
flatten_15/Const?
flatten_15/ReshapeReshapedropout_61/dropout/Mul_1:z:0flatten_15/Const:output:0*
T0*(
_output_shapes
:??????????Q2
flatten_15/Reshape?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
?Q?*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMulflatten_15/Reshape:output:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/BiasAddt
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_36/Relu?
5batch_normalization_46/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_46/moments/mean/reduction_indices?
#batch_normalization_46/moments/meanMeandense_36/Relu:activations:0>batch_normalization_46/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2%
#batch_normalization_46/moments/mean?
+batch_normalization_46/moments/StopGradientStopGradient,batch_normalization_46/moments/mean:output:0*
T0*
_output_shapes
:	?2-
+batch_normalization_46/moments/StopGradient?
0batch_normalization_46/moments/SquaredDifferenceSquaredDifferencedense_36/Relu:activations:04batch_normalization_46/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????22
0batch_normalization_46/moments/SquaredDifference?
9batch_normalization_46/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_46/moments/variance/reduction_indices?
'batch_normalization_46/moments/varianceMean4batch_normalization_46/moments/SquaredDifference:z:0Bbatch_normalization_46/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2)
'batch_normalization_46/moments/variance?
&batch_normalization_46/moments/SqueezeSqueeze,batch_normalization_46/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_46/moments/Squeeze?
(batch_normalization_46/moments/Squeeze_1Squeeze0batch_normalization_46/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_46/moments/Squeeze_1?
,batch_normalization_46/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_46/AssignMovingAvg/671794*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_46/AssignMovingAvg/decay?
5batch_normalization_46/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_46_assignmovingavg_671794*
_output_shapes	
:?*
dtype027
5batch_normalization_46/AssignMovingAvg/ReadVariableOp?
*batch_normalization_46/AssignMovingAvg/subSub=batch_normalization_46/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_46/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_46/AssignMovingAvg/671794*
_output_shapes	
:?2,
*batch_normalization_46/AssignMovingAvg/sub?
*batch_normalization_46/AssignMovingAvg/mulMul.batch_normalization_46/AssignMovingAvg/sub:z:05batch_normalization_46/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_46/AssignMovingAvg/671794*
_output_shapes	
:?2,
*batch_normalization_46/AssignMovingAvg/mul?
:batch_normalization_46/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_46_assignmovingavg_671794.batch_normalization_46/AssignMovingAvg/mul:z:06^batch_normalization_46/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_46/AssignMovingAvg/671794*
_output_shapes
 *
dtype02<
:batch_normalization_46/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_46/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_46/AssignMovingAvg_1/671800*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_46/AssignMovingAvg_1/decay?
7batch_normalization_46/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_46_assignmovingavg_1_671800*
_output_shapes	
:?*
dtype029
7batch_normalization_46/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_46/AssignMovingAvg_1/subSub?batch_normalization_46/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_46/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_46/AssignMovingAvg_1/671800*
_output_shapes	
:?2.
,batch_normalization_46/AssignMovingAvg_1/sub?
,batch_normalization_46/AssignMovingAvg_1/mulMul0batch_normalization_46/AssignMovingAvg_1/sub:z:07batch_normalization_46/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_46/AssignMovingAvg_1/671800*
_output_shapes	
:?2.
,batch_normalization_46/AssignMovingAvg_1/mul?
<batch_normalization_46/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_46_assignmovingavg_1_6718000batch_normalization_46/AssignMovingAvg_1/mul:z:08^batch_normalization_46/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_46/AssignMovingAvg_1/671800*
_output_shapes
 *
dtype02>
<batch_normalization_46/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_46/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_46/batchnorm/add/y?
$batch_normalization_46/batchnorm/addAddV21batch_normalization_46/moments/Squeeze_1:output:0/batch_normalization_46/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_46/batchnorm/add?
&batch_normalization_46/batchnorm/RsqrtRsqrt(batch_normalization_46/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_46/batchnorm/Rsqrt?
3batch_normalization_46/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_46_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_46/batchnorm/mul/ReadVariableOp?
$batch_normalization_46/batchnorm/mulMul*batch_normalization_46/batchnorm/Rsqrt:y:0;batch_normalization_46/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_46/batchnorm/mul?
&batch_normalization_46/batchnorm/mul_1Muldense_36/Relu:activations:0(batch_normalization_46/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_46/batchnorm/mul_1?
&batch_normalization_46/batchnorm/mul_2Mul/batch_normalization_46/moments/Squeeze:output:0(batch_normalization_46/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_46/batchnorm/mul_2?
/batch_normalization_46/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_46_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_46/batchnorm/ReadVariableOp?
$batch_normalization_46/batchnorm/subSub7batch_normalization_46/batchnorm/ReadVariableOp:value:0*batch_normalization_46/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_46/batchnorm/sub?
&batch_normalization_46/batchnorm/add_1AddV2*batch_normalization_46/batchnorm/mul_1:z:0(batch_normalization_46/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_46/batchnorm/add_1y
dropout_62/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_62/dropout/Const?
dropout_62/dropout/MulMul*batch_normalization_46/batchnorm/add_1:z:0!dropout_62/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_62/dropout/Mul?
dropout_62/dropout/ShapeShape*batch_normalization_46/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_62/dropout/Shape?
/dropout_62/dropout/random_uniform/RandomUniformRandomUniform!dropout_62/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed?*
seed221
/dropout_62/dropout/random_uniform/RandomUniform?
!dropout_62/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_62/dropout/GreaterEqual/y?
dropout_62/dropout/GreaterEqualGreaterEqual8dropout_62/dropout/random_uniform/RandomUniform:output:0*dropout_62/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
dropout_62/dropout/GreaterEqual?
dropout_62/dropout/CastCast#dropout_62/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_62/dropout/Cast?
dropout_62/dropout/Mul_1Muldropout_62/dropout/Mul:z:0dropout_62/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_62/dropout/Mul_1?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldropout_62/dropout/Mul_1:z:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_37/BiasAdd|
dense_37/SoftmaxSoftmaxdense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_37/Softmax?
IdentityIdentitydense_37/Softmax:softmax:0&^batch_normalization_43/AssignNewValue(^batch_normalization_43/AssignNewValue_17^batch_normalization_43/FusedBatchNormV3/ReadVariableOp9^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_43/ReadVariableOp(^batch_normalization_43/ReadVariableOp_1&^batch_normalization_44/AssignNewValue(^batch_normalization_44/AssignNewValue_17^batch_normalization_44/FusedBatchNormV3/ReadVariableOp9^batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_44/ReadVariableOp(^batch_normalization_44/ReadVariableOp_1&^batch_normalization_45/AssignNewValue(^batch_normalization_45/AssignNewValue_17^batch_normalization_45/FusedBatchNormV3/ReadVariableOp9^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_45/ReadVariableOp(^batch_normalization_45/ReadVariableOp_1;^batch_normalization_46/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_46/AssignMovingAvg/ReadVariableOp=^batch_normalization_46/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_46/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_46/batchnorm/ReadVariableOp4^batch_normalization_46/batchnorm/mul/ReadVariableOp!^conv2d_57/BiasAdd/ReadVariableOp ^conv2d_57/Conv2D/ReadVariableOp!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::2N
%batch_normalization_43/AssignNewValue%batch_normalization_43/AssignNewValue2R
'batch_normalization_43/AssignNewValue_1'batch_normalization_43/AssignNewValue_12p
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp6batch_normalization_43/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_18batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_43/ReadVariableOp%batch_normalization_43/ReadVariableOp2R
'batch_normalization_43/ReadVariableOp_1'batch_normalization_43/ReadVariableOp_12N
%batch_normalization_44/AssignNewValue%batch_normalization_44/AssignNewValue2R
'batch_normalization_44/AssignNewValue_1'batch_normalization_44/AssignNewValue_12p
6batch_normalization_44/FusedBatchNormV3/ReadVariableOp6batch_normalization_44/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_44/FusedBatchNormV3/ReadVariableOp_18batch_normalization_44/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_44/ReadVariableOp%batch_normalization_44/ReadVariableOp2R
'batch_normalization_44/ReadVariableOp_1'batch_normalization_44/ReadVariableOp_12N
%batch_normalization_45/AssignNewValue%batch_normalization_45/AssignNewValue2R
'batch_normalization_45/AssignNewValue_1'batch_normalization_45/AssignNewValue_12p
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp6batch_normalization_45/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_18batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_45/ReadVariableOp%batch_normalization_45/ReadVariableOp2R
'batch_normalization_45/ReadVariableOp_1'batch_normalization_45/ReadVariableOp_12x
:batch_normalization_46/AssignMovingAvg/AssignSubVariableOp:batch_normalization_46/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_46/AssignMovingAvg/ReadVariableOp5batch_normalization_46/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_46/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_46/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_46/AssignMovingAvg_1/ReadVariableOp7batch_normalization_46/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_46/batchnorm/ReadVariableOp/batch_normalization_46/batchnorm/ReadVariableOp2j
3batch_normalization_46/batchnorm/mul/ReadVariableOp3batch_normalization_46/batchnorm/mul/ReadVariableOp2D
 conv2d_57/BiasAdd/ReadVariableOp conv2d_57/BiasAdd/ReadVariableOp2B
conv2d_57/Conv2D/ReadVariableOpconv2d_57/Conv2D/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_59_layer_call_and_return_conditional_losses_672401

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

*__inference_conv2d_58_layer_call_fn_672235

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_58_layer_call_and_return_conditional_losses_6708182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_672477

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????		?::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_45_layer_call_fn_672572

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_6705212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_17_layer_call_fn_672067

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_6715512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_670521

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_59_layer_call_and_return_conditional_losses_670948

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
F__inference_dropout_59_layer_call_and_return_conditional_losses_672380

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_670552

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_60_layer_call_and_return_conditional_losses_672448

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_672699

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_61_layer_call_fn_672612

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_61_layer_call_and_return_conditional_losses_6711122
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
e
F__inference_dropout_60_layer_call_and_return_conditional_losses_672422

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?0
?
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_672679

inputs
assignmovingavg_672654
assignmovingavg_1_672660)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/672654*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_672654*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/672654*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg/672654*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_672654AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg/672654*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/672660*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_672660*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/672660*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/672660*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_672660AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg_1/672660*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_36_layer_call_and_return_conditional_losses_671150

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?Q?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????Q::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????Q
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_43_layer_call_fn_672202

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_6703012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_670239
conv2d_57_input:
6sequential_17_conv2d_57_conv2d_readvariableop_resource;
7sequential_17_conv2d_57_biasadd_readvariableop_resource@
<sequential_17_batch_normalization_43_readvariableop_resourceB
>sequential_17_batch_normalization_43_readvariableop_1_resourceQ
Msequential_17_batch_normalization_43_fusedbatchnormv3_readvariableop_resourceS
Osequential_17_batch_normalization_43_fusedbatchnormv3_readvariableop_1_resource:
6sequential_17_conv2d_58_conv2d_readvariableop_resource;
7sequential_17_conv2d_58_biasadd_readvariableop_resource@
<sequential_17_batch_normalization_44_readvariableop_resourceB
>sequential_17_batch_normalization_44_readvariableop_1_resourceQ
Msequential_17_batch_normalization_44_fusedbatchnormv3_readvariableop_resourceS
Osequential_17_batch_normalization_44_fusedbatchnormv3_readvariableop_1_resource:
6sequential_17_conv2d_59_conv2d_readvariableop_resource;
7sequential_17_conv2d_59_biasadd_readvariableop_resource:
6sequential_17_conv2d_60_conv2d_readvariableop_resource;
7sequential_17_conv2d_60_biasadd_readvariableop_resource@
<sequential_17_batch_normalization_45_readvariableop_resourceB
>sequential_17_batch_normalization_45_readvariableop_1_resourceQ
Msequential_17_batch_normalization_45_fusedbatchnormv3_readvariableop_resourceS
Osequential_17_batch_normalization_45_fusedbatchnormv3_readvariableop_1_resource9
5sequential_17_dense_36_matmul_readvariableop_resource:
6sequential_17_dense_36_biasadd_readvariableop_resourceJ
Fsequential_17_batch_normalization_46_batchnorm_readvariableop_resourceN
Jsequential_17_batch_normalization_46_batchnorm_mul_readvariableop_resourceL
Hsequential_17_batch_normalization_46_batchnorm_readvariableop_1_resourceL
Hsequential_17_batch_normalization_46_batchnorm_readvariableop_2_resource9
5sequential_17_dense_37_matmul_readvariableop_resource:
6sequential_17_dense_37_biasadd_readvariableop_resource
identity??Dsequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOp?Fsequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?3sequential_17/batch_normalization_43/ReadVariableOp?5sequential_17/batch_normalization_43/ReadVariableOp_1?Dsequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOp?Fsequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?3sequential_17/batch_normalization_44/ReadVariableOp?5sequential_17/batch_normalization_44/ReadVariableOp_1?Dsequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOp?Fsequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?3sequential_17/batch_normalization_45/ReadVariableOp?5sequential_17/batch_normalization_45/ReadVariableOp_1?=sequential_17/batch_normalization_46/batchnorm/ReadVariableOp??sequential_17/batch_normalization_46/batchnorm/ReadVariableOp_1??sequential_17/batch_normalization_46/batchnorm/ReadVariableOp_2?Asequential_17/batch_normalization_46/batchnorm/mul/ReadVariableOp?.sequential_17/conv2d_57/BiasAdd/ReadVariableOp?-sequential_17/conv2d_57/Conv2D/ReadVariableOp?.sequential_17/conv2d_58/BiasAdd/ReadVariableOp?-sequential_17/conv2d_58/Conv2D/ReadVariableOp?.sequential_17/conv2d_59/BiasAdd/ReadVariableOp?-sequential_17/conv2d_59/Conv2D/ReadVariableOp?.sequential_17/conv2d_60/BiasAdd/ReadVariableOp?-sequential_17/conv2d_60/Conv2D/ReadVariableOp?-sequential_17/dense_36/BiasAdd/ReadVariableOp?,sequential_17/dense_36/MatMul/ReadVariableOp?-sequential_17/dense_37/BiasAdd/ReadVariableOp?,sequential_17/dense_37/MatMul/ReadVariableOp?
-sequential_17/conv2d_57/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_57_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_17/conv2d_57/Conv2D/ReadVariableOp?
sequential_17/conv2d_57/Conv2DConv2Dconv2d_57_input5sequential_17/conv2d_57/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2 
sequential_17/conv2d_57/Conv2D?
.sequential_17/conv2d_57/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_57_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_57/BiasAdd/ReadVariableOp?
sequential_17/conv2d_57/BiasAddBiasAdd'sequential_17/conv2d_57/Conv2D:output:06sequential_17/conv2d_57/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2!
sequential_17/conv2d_57/BiasAdd?
sequential_17/conv2d_57/ReluRelu(sequential_17/conv2d_57/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_17/conv2d_57/Relu?
3sequential_17/batch_normalization_43/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_43_readvariableop_resource*
_output_shapes
: *
dtype025
3sequential_17/batch_normalization_43/ReadVariableOp?
5sequential_17/batch_normalization_43/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_43_readvariableop_1_resource*
_output_shapes
: *
dtype027
5sequential_17/batch_normalization_43/ReadVariableOp_1?
Dsequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dsequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOp?
Fsequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fsequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?
5sequential_17/batch_normalization_43/FusedBatchNormV3FusedBatchNormV3*sequential_17/conv2d_57/Relu:activations:0;sequential_17/batch_normalization_43/ReadVariableOp:value:0=sequential_17/batch_normalization_43/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 27
5sequential_17/batch_normalization_43/FusedBatchNormV3?
-sequential_17/conv2d_58/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02/
-sequential_17/conv2d_58/Conv2D/ReadVariableOp?
sequential_17/conv2d_58/Conv2DConv2D9sequential_17/batch_normalization_43/FusedBatchNormV3:y:05sequential_17/conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2 
sequential_17/conv2d_58/Conv2D?
.sequential_17/conv2d_58/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_58/BiasAdd/ReadVariableOp?
sequential_17/conv2d_58/BiasAddBiasAdd'sequential_17/conv2d_58/Conv2D:output:06sequential_17/conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2!
sequential_17/conv2d_58/BiasAdd?
sequential_17/conv2d_58/ReluRelu(sequential_17/conv2d_58/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_17/conv2d_58/Relu?
3sequential_17/batch_normalization_44/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_44_readvariableop_resource*
_output_shapes
: *
dtype025
3sequential_17/batch_normalization_44/ReadVariableOp?
5sequential_17/batch_normalization_44/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_44_readvariableop_1_resource*
_output_shapes
: *
dtype027
5sequential_17/batch_normalization_44/ReadVariableOp_1?
Dsequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_44_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dsequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOp?
Fsequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_44_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fsequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1?
5sequential_17/batch_normalization_44/FusedBatchNormV3FusedBatchNormV3*sequential_17/conv2d_58/Relu:activations:0;sequential_17/batch_normalization_44/ReadVariableOp:value:0=sequential_17/batch_normalization_44/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 27
5sequential_17/batch_normalization_44/FusedBatchNormV3?
!sequential_17/dropout_59/IdentityIdentity9sequential_17/batch_normalization_44/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2#
!sequential_17/dropout_59/Identity?
-sequential_17/conv2d_59/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_17/conv2d_59/Conv2D/ReadVariableOp?
sequential_17/conv2d_59/Conv2DConv2D*sequential_17/dropout_59/Identity:output:05sequential_17/conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2 
sequential_17/conv2d_59/Conv2D?
.sequential_17/conv2d_59/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_59_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_17/conv2d_59/BiasAdd/ReadVariableOp?
sequential_17/conv2d_59/BiasAddBiasAdd'sequential_17/conv2d_59/Conv2D:output:06sequential_17/conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2!
sequential_17/conv2d_59/BiasAdd?
sequential_17/conv2d_59/ReluRelu(sequential_17/conv2d_59/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_17/conv2d_59/Relu?
&sequential_17/max_pooling2d_23/MaxPoolMaxPool*sequential_17/conv2d_59/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_23/MaxPool?
!sequential_17/dropout_60/IdentityIdentity/sequential_17/max_pooling2d_23/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2#
!sequential_17/dropout_60/Identity?
-sequential_17/conv2d_60/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_60_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02/
-sequential_17/conv2d_60/Conv2D/ReadVariableOp?
sequential_17/conv2d_60/Conv2DConv2D*sequential_17/dropout_60/Identity:output:05sequential_17/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2 
sequential_17/conv2d_60/Conv2D?
.sequential_17/conv2d_60/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_60_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_17/conv2d_60/BiasAdd/ReadVariableOp?
sequential_17/conv2d_60/BiasAddBiasAdd'sequential_17/conv2d_60/Conv2D:output:06sequential_17/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2!
sequential_17/conv2d_60/BiasAdd?
sequential_17/conv2d_60/ReluRelu(sequential_17/conv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
sequential_17/conv2d_60/Relu?
3sequential_17/batch_normalization_45/ReadVariableOpReadVariableOp<sequential_17_batch_normalization_45_readvariableop_resource*
_output_shapes	
:?*
dtype025
3sequential_17/batch_normalization_45/ReadVariableOp?
5sequential_17/batch_normalization_45/ReadVariableOp_1ReadVariableOp>sequential_17_batch_normalization_45_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5sequential_17/batch_normalization_45/ReadVariableOp_1?
Dsequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_17_batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02F
Dsequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOp?
Fsequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_17_batch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02H
Fsequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?
5sequential_17/batch_normalization_45/FusedBatchNormV3FusedBatchNormV3*sequential_17/conv2d_60/Relu:activations:0;sequential_17/batch_normalization_45/ReadVariableOp:value:0=sequential_17/batch_normalization_45/ReadVariableOp_1:value:0Lsequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
is_training( 27
5sequential_17/batch_normalization_45/FusedBatchNormV3?
!sequential_17/dropout_61/IdentityIdentity9sequential_17/batch_normalization_45/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????		?2#
!sequential_17/dropout_61/Identity?
sequential_17/flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????(  2 
sequential_17/flatten_15/Const?
 sequential_17/flatten_15/ReshapeReshape*sequential_17/dropout_61/Identity:output:0'sequential_17/flatten_15/Const:output:0*
T0*(
_output_shapes
:??????????Q2"
 sequential_17/flatten_15/Reshape?
,sequential_17/dense_36/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
?Q?*
dtype02.
,sequential_17/dense_36/MatMul/ReadVariableOp?
sequential_17/dense_36/MatMulMatMul)sequential_17/flatten_15/Reshape:output:04sequential_17/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_17/dense_36/MatMul?
-sequential_17/dense_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_17/dense_36/BiasAdd/ReadVariableOp?
sequential_17/dense_36/BiasAddBiasAdd'sequential_17/dense_36/MatMul:product:05sequential_17/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_17/dense_36/BiasAdd?
sequential_17/dense_36/ReluRelu'sequential_17/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_17/dense_36/Relu?
=sequential_17/batch_normalization_46/batchnorm/ReadVariableOpReadVariableOpFsequential_17_batch_normalization_46_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=sequential_17/batch_normalization_46/batchnorm/ReadVariableOp?
4sequential_17/batch_normalization_46/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:26
4sequential_17/batch_normalization_46/batchnorm/add/y?
2sequential_17/batch_normalization_46/batchnorm/addAddV2Esequential_17/batch_normalization_46/batchnorm/ReadVariableOp:value:0=sequential_17/batch_normalization_46/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?24
2sequential_17/batch_normalization_46/batchnorm/add?
4sequential_17/batch_normalization_46/batchnorm/RsqrtRsqrt6sequential_17/batch_normalization_46/batchnorm/add:z:0*
T0*
_output_shapes	
:?26
4sequential_17/batch_normalization_46/batchnorm/Rsqrt?
Asequential_17/batch_normalization_46/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_17_batch_normalization_46_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Asequential_17/batch_normalization_46/batchnorm/mul/ReadVariableOp?
2sequential_17/batch_normalization_46/batchnorm/mulMul8sequential_17/batch_normalization_46/batchnorm/Rsqrt:y:0Isequential_17/batch_normalization_46/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?24
2sequential_17/batch_normalization_46/batchnorm/mul?
4sequential_17/batch_normalization_46/batchnorm/mul_1Mul)sequential_17/dense_36/Relu:activations:06sequential_17/batch_normalization_46/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????26
4sequential_17/batch_normalization_46/batchnorm/mul_1?
?sequential_17/batch_normalization_46/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_17_batch_normalization_46_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02A
?sequential_17/batch_normalization_46/batchnorm/ReadVariableOp_1?
4sequential_17/batch_normalization_46/batchnorm/mul_2MulGsequential_17/batch_normalization_46/batchnorm/ReadVariableOp_1:value:06sequential_17/batch_normalization_46/batchnorm/mul:z:0*
T0*
_output_shapes	
:?26
4sequential_17/batch_normalization_46/batchnorm/mul_2?
?sequential_17/batch_normalization_46/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_17_batch_normalization_46_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02A
?sequential_17/batch_normalization_46/batchnorm/ReadVariableOp_2?
2sequential_17/batch_normalization_46/batchnorm/subSubGsequential_17/batch_normalization_46/batchnorm/ReadVariableOp_2:value:08sequential_17/batch_normalization_46/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?24
2sequential_17/batch_normalization_46/batchnorm/sub?
4sequential_17/batch_normalization_46/batchnorm/add_1AddV28sequential_17/batch_normalization_46/batchnorm/mul_1:z:06sequential_17/batch_normalization_46/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????26
4sequential_17/batch_normalization_46/batchnorm/add_1?
!sequential_17/dropout_62/IdentityIdentity8sequential_17/batch_normalization_46/batchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!sequential_17/dropout_62/Identity?
,sequential_17/dense_37/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02.
,sequential_17/dense_37/MatMul/ReadVariableOp?
sequential_17/dense_37/MatMulMatMul*sequential_17/dropout_62/Identity:output:04sequential_17/dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_17/dense_37/MatMul?
-sequential_17/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_17/dense_37/BiasAdd/ReadVariableOp?
sequential_17/dense_37/BiasAddBiasAdd'sequential_17/dense_37/MatMul:product:05sequential_17/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
sequential_17/dense_37/BiasAdd?
sequential_17/dense_37/SoftmaxSoftmax'sequential_17/dense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2 
sequential_17/dense_37/Softmax?
IdentityIdentity(sequential_17/dense_37/Softmax:softmax:0E^sequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_43/ReadVariableOp6^sequential_17/batch_normalization_43/ReadVariableOp_1E^sequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_44/ReadVariableOp6^sequential_17/batch_normalization_44/ReadVariableOp_1E^sequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOpG^sequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_14^sequential_17/batch_normalization_45/ReadVariableOp6^sequential_17/batch_normalization_45/ReadVariableOp_1>^sequential_17/batch_normalization_46/batchnorm/ReadVariableOp@^sequential_17/batch_normalization_46/batchnorm/ReadVariableOp_1@^sequential_17/batch_normalization_46/batchnorm/ReadVariableOp_2B^sequential_17/batch_normalization_46/batchnorm/mul/ReadVariableOp/^sequential_17/conv2d_57/BiasAdd/ReadVariableOp.^sequential_17/conv2d_57/Conv2D/ReadVariableOp/^sequential_17/conv2d_58/BiasAdd/ReadVariableOp.^sequential_17/conv2d_58/Conv2D/ReadVariableOp/^sequential_17/conv2d_59/BiasAdd/ReadVariableOp.^sequential_17/conv2d_59/Conv2D/ReadVariableOp/^sequential_17/conv2d_60/BiasAdd/ReadVariableOp.^sequential_17/conv2d_60/Conv2D/ReadVariableOp.^sequential_17/dense_36/BiasAdd/ReadVariableOp-^sequential_17/dense_36/MatMul/ReadVariableOp.^sequential_17/dense_37/BiasAdd/ReadVariableOp-^sequential_17/dense_37/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::2?
Dsequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOp2?
Fsequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_43/ReadVariableOp3sequential_17/batch_normalization_43/ReadVariableOp2n
5sequential_17/batch_normalization_43/ReadVariableOp_15sequential_17/batch_normalization_43/ReadVariableOp_12?
Dsequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOp2?
Fsequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_44/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_44/ReadVariableOp3sequential_17/batch_normalization_44/ReadVariableOp2n
5sequential_17/batch_normalization_44/ReadVariableOp_15sequential_17/batch_normalization_44/ReadVariableOp_12?
Dsequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOpDsequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOp2?
Fsequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1Fsequential_17/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12j
3sequential_17/batch_normalization_45/ReadVariableOp3sequential_17/batch_normalization_45/ReadVariableOp2n
5sequential_17/batch_normalization_45/ReadVariableOp_15sequential_17/batch_normalization_45/ReadVariableOp_12~
=sequential_17/batch_normalization_46/batchnorm/ReadVariableOp=sequential_17/batch_normalization_46/batchnorm/ReadVariableOp2?
?sequential_17/batch_normalization_46/batchnorm/ReadVariableOp_1?sequential_17/batch_normalization_46/batchnorm/ReadVariableOp_12?
?sequential_17/batch_normalization_46/batchnorm/ReadVariableOp_2?sequential_17/batch_normalization_46/batchnorm/ReadVariableOp_22?
Asequential_17/batch_normalization_46/batchnorm/mul/ReadVariableOpAsequential_17/batch_normalization_46/batchnorm/mul/ReadVariableOp2`
.sequential_17/conv2d_57/BiasAdd/ReadVariableOp.sequential_17/conv2d_57/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_57/Conv2D/ReadVariableOp-sequential_17/conv2d_57/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_58/BiasAdd/ReadVariableOp.sequential_17/conv2d_58/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_58/Conv2D/ReadVariableOp-sequential_17/conv2d_58/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_59/BiasAdd/ReadVariableOp.sequential_17/conv2d_59/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_59/Conv2D/ReadVariableOp-sequential_17/conv2d_59/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_60/BiasAdd/ReadVariableOp.sequential_17/conv2d_60/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_60/Conv2D/ReadVariableOp-sequential_17/conv2d_60/Conv2D/ReadVariableOp2^
-sequential_17/dense_36/BiasAdd/ReadVariableOp-sequential_17/dense_36/BiasAdd/ReadVariableOp2\
,sequential_17/dense_36/MatMul/ReadVariableOp,sequential_17/dense_36/MatMul/ReadVariableOp2^
-sequential_17/dense_37/BiasAdd/ReadVariableOp-sequential_17/dense_37/BiasAdd/ReadVariableOp2\
,sequential_17/dense_37/MatMul/ReadVariableOp,sequential_17/dense_37/MatMul/ReadVariableOp:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_57_input
?	
?
D__inference_dense_36_layer_call_and_return_conditional_losses_672634

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?Q?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????Q::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????Q
 
_user_specified_nameinputs
?
e
F__inference_dropout_62_layer_call_and_return_conditional_losses_671213

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_dense_37_layer_call_fn_672772

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_6712422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_58_layer_call_and_return_conditional_losses_670818

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_672107

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
G
+__inference_dropout_59_layer_call_fn_672390

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_59_layer_call_and_return_conditional_losses_6709242
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_46_layer_call_fn_672725

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_6706922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_61_layer_call_and_return_conditional_losses_672602

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????		?2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????		?2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_44_layer_call_fn_672286

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_6704052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
d
F__inference_dropout_62_layer_call_and_return_conditional_losses_672742

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_672495

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????		?::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
e
F__inference_dropout_60_layer_call_and_return_conditional_losses_670977

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
e
F__inference_dropout_62_layer_call_and_return_conditional_losses_672737

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?W
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_671259
conv2d_57_input
conv2d_57_670729
conv2d_57_670731!
batch_normalization_43_670798!
batch_normalization_43_670800!
batch_normalization_43_670802!
batch_normalization_43_670804
conv2d_58_670829
conv2d_58_670831!
batch_normalization_44_670898!
batch_normalization_44_670900!
batch_normalization_44_670902!
batch_normalization_44_670904
conv2d_59_670959
conv2d_59_670961
conv2d_60_671017
conv2d_60_671019!
batch_normalization_45_671086!
batch_normalization_45_671088!
batch_normalization_45_671090!
batch_normalization_45_671092
dense_36_671161
dense_36_671163!
batch_normalization_46_671192!
batch_normalization_46_671194!
batch_normalization_46_671196!
batch_normalization_46_671198
dense_37_671253
dense_37_671255
identity??.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_46/StatefulPartitionedCall?!conv2d_57/StatefulPartitionedCall?!conv2d_58/StatefulPartitionedCall?!conv2d_59/StatefulPartitionedCall?!conv2d_60/StatefulPartitionedCall? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?"dropout_59/StatefulPartitionedCall?"dropout_60/StatefulPartitionedCall?"dropout_61/StatefulPartitionedCall?"dropout_62/StatefulPartitionedCall?
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCallconv2d_57_inputconv2d_57_670729conv2d_57_670731*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_57_layer_call_and_return_conditional_losses_6707182#
!conv2d_57/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0batch_normalization_43_670798batch_normalization_43_670800batch_normalization_43_670802batch_normalization_43_670804*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_67075320
.batch_normalization_43/StatefulPartitionedCall?
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0conv2d_58_670829conv2d_58_670831*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_58_layer_call_and_return_conditional_losses_6708182#
!conv2d_58/StatefulPartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_44_670898batch_normalization_44_670900batch_normalization_44_670902batch_normalization_44_670904*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_67085320
.batch_normalization_44/StatefulPartitionedCall?
"dropout_59/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_59_layer_call_and_return_conditional_losses_6709192$
"dropout_59/StatefulPartitionedCall?
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall+dropout_59/StatefulPartitionedCall:output:0conv2d_59_670959conv2d_59_670961*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_59_layer_call_and_return_conditional_losses_6709482#
!conv2d_59/StatefulPartitionedCall?
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_6704532"
 max_pooling2d_23/PartitionedCall?
"dropout_60/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_23/PartitionedCall:output:0#^dropout_59/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_60_layer_call_and_return_conditional_losses_6709772$
"dropout_60/StatefulPartitionedCall?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall+dropout_60/StatefulPartitionedCall:output:0conv2d_60_671017conv2d_60_671019*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_60_layer_call_and_return_conditional_losses_6710062#
!conv2d_60/StatefulPartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0batch_normalization_45_671086batch_normalization_45_671088batch_normalization_45_671090batch_normalization_45_671092*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_67104120
.batch_normalization_45/StatefulPartitionedCall?
"dropout_61/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0#^dropout_60/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_61_layer_call_and_return_conditional_losses_6711072$
"dropout_61/StatefulPartitionedCall?
flatten_15/PartitionedCallPartitionedCall+dropout_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_6711312
flatten_15/PartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_36_671161dense_36_671163*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_6711502"
 dense_36/StatefulPartitionedCall?
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0batch_normalization_46_671192batch_normalization_46_671194batch_normalization_46_671196batch_normalization_46_671198*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_67065920
.batch_normalization_46/StatefulPartitionedCall?
"dropout_62/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_46/StatefulPartitionedCall:output:0#^dropout_61/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_62_layer_call_and_return_conditional_losses_6712132$
"dropout_62/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall+dropout_62/StatefulPartitionedCall:output:0dense_37_671253dense_37_671255*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_6712422"
 dense_37/StatefulPartitionedCall?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_46/StatefulPartitionedCall"^conv2d_57/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall#^dropout_59/StatefulPartitionedCall#^dropout_60/StatefulPartitionedCall#^dropout_61/StatefulPartitionedCall#^dropout_62/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2H
"dropout_59/StatefulPartitionedCall"dropout_59/StatefulPartitionedCall2H
"dropout_60/StatefulPartitionedCall"dropout_60/StatefulPartitionedCall2H
"dropout_61/StatefulPartitionedCall"dropout_61/StatefulPartitionedCall2H
"dropout_62/StatefulPartitionedCall"dropout_62/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_57_input
?
?
.__inference_sequential_17_layer_call_fn_671473
conv2d_57_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_57_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_6714142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_57_input
?
d
+__inference_dropout_60_layer_call_fn_672432

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_60_layer_call_and_return_conditional_losses_6709772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_670436

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
E__inference_conv2d_58_layer_call_and_return_conditional_losses_672226

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_43_layer_call_fn_672138

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_6707532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_671041

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????		?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????		?::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_44_layer_call_fn_672363

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_6708712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?G
?
__inference__traced_save_672891
file_prefix/
+savev2_conv2d_57_kernel_read_readvariableop-
)savev2_conv2d_57_bias_read_readvariableop;
7savev2_batch_normalization_43_gamma_read_readvariableop:
6savev2_batch_normalization_43_beta_read_readvariableopA
=savev2_batch_normalization_43_moving_mean_read_readvariableopE
Asavev2_batch_normalization_43_moving_variance_read_readvariableop/
+savev2_conv2d_58_kernel_read_readvariableop-
)savev2_conv2d_58_bias_read_readvariableop;
7savev2_batch_normalization_44_gamma_read_readvariableop:
6savev2_batch_normalization_44_beta_read_readvariableopA
=savev2_batch_normalization_44_moving_mean_read_readvariableopE
Asavev2_batch_normalization_44_moving_variance_read_readvariableop/
+savev2_conv2d_59_kernel_read_readvariableop-
)savev2_conv2d_59_bias_read_readvariableop/
+savev2_conv2d_60_kernel_read_readvariableop-
)savev2_conv2d_60_bias_read_readvariableop;
7savev2_batch_normalization_45_gamma_read_readvariableop:
6savev2_batch_normalization_45_beta_read_readvariableopA
=savev2_batch_normalization_45_moving_mean_read_readvariableopE
Asavev2_batch_normalization_45_moving_variance_read_readvariableop.
*savev2_dense_36_kernel_read_readvariableop,
(savev2_dense_36_bias_read_readvariableop;
7savev2_batch_normalization_46_gamma_read_readvariableop:
6savev2_batch_normalization_46_beta_read_readvariableopA
=savev2_batch_normalization_46_moving_mean_read_readvariableopE
Asavev2_batch_normalization_46_moving_variance_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_57_kernel_read_readvariableop)savev2_conv2d_57_bias_read_readvariableop7savev2_batch_normalization_43_gamma_read_readvariableop6savev2_batch_normalization_43_beta_read_readvariableop=savev2_batch_normalization_43_moving_mean_read_readvariableopAsavev2_batch_normalization_43_moving_variance_read_readvariableop+savev2_conv2d_58_kernel_read_readvariableop)savev2_conv2d_58_bias_read_readvariableop7savev2_batch_normalization_44_gamma_read_readvariableop6savev2_batch_normalization_44_beta_read_readvariableop=savev2_batch_normalization_44_moving_mean_read_readvariableopAsavev2_batch_normalization_44_moving_variance_read_readvariableop+savev2_conv2d_59_kernel_read_readvariableop)savev2_conv2d_59_bias_read_readvariableop+savev2_conv2d_60_kernel_read_readvariableop)savev2_conv2d_60_bias_read_readvariableop7savev2_batch_normalization_45_gamma_read_readvariableop6savev2_batch_normalization_45_beta_read_readvariableop=savev2_batch_normalization_45_moving_mean_read_readvariableopAsavev2_batch_normalization_45_moving_variance_read_readvariableop*savev2_dense_36_kernel_read_readvariableop(savev2_dense_36_bias_read_readvariableop7savev2_batch_normalization_46_gamma_read_readvariableop6savev2_batch_normalization_46_beta_read_readvariableop=savev2_batch_normalization_46_moving_mean_read_readvariableopAsavev2_batch_normalization_46_moving_variance_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : :  : : : : : : @:@:@?:?:?:?:?:?:
?Q?:?:?:?:?:?:	?
:
: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
?Q?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: 
?	
?
D__inference_dense_37_layer_call_and_return_conditional_losses_672763

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_59_layer_call_and_return_conditional_losses_670919

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?Q
?

I__inference_sequential_17_layer_call_and_return_conditional_losses_671335
conv2d_57_input
conv2d_57_671262
conv2d_57_671264!
batch_normalization_43_671267!
batch_normalization_43_671269!
batch_normalization_43_671271!
batch_normalization_43_671273
conv2d_58_671276
conv2d_58_671278!
batch_normalization_44_671281!
batch_normalization_44_671283!
batch_normalization_44_671285!
batch_normalization_44_671287
conv2d_59_671291
conv2d_59_671293
conv2d_60_671298
conv2d_60_671300!
batch_normalization_45_671303!
batch_normalization_45_671305!
batch_normalization_45_671307!
batch_normalization_45_671309
dense_36_671314
dense_36_671316!
batch_normalization_46_671319!
batch_normalization_46_671321!
batch_normalization_46_671323!
batch_normalization_46_671325
dense_37_671329
dense_37_671331
identity??.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_44/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_46/StatefulPartitionedCall?!conv2d_57/StatefulPartitionedCall?!conv2d_58/StatefulPartitionedCall?!conv2d_59/StatefulPartitionedCall?!conv2d_60/StatefulPartitionedCall? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCallconv2d_57_inputconv2d_57_671262conv2d_57_671264*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_57_layer_call_and_return_conditional_losses_6707182#
!conv2d_57/StatefulPartitionedCall?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0batch_normalization_43_671267batch_normalization_43_671269batch_normalization_43_671271batch_normalization_43_671273*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_67077120
.batch_normalization_43/StatefulPartitionedCall?
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0conv2d_58_671276conv2d_58_671278*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_58_layer_call_and_return_conditional_losses_6708182#
!conv2d_58/StatefulPartitionedCall?
.batch_normalization_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0batch_normalization_44_671281batch_normalization_44_671283batch_normalization_44_671285batch_normalization_44_671287*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_67087120
.batch_normalization_44/StatefulPartitionedCall?
dropout_59/PartitionedCallPartitionedCall7batch_normalization_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_59_layer_call_and_return_conditional_losses_6709242
dropout_59/PartitionedCall?
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall#dropout_59/PartitionedCall:output:0conv2d_59_671291conv2d_59_671293*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_59_layer_call_and_return_conditional_losses_6709482#
!conv2d_59/StatefulPartitionedCall?
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_6704532"
 max_pooling2d_23/PartitionedCall?
dropout_60/PartitionedCallPartitionedCall)max_pooling2d_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_60_layer_call_and_return_conditional_losses_6709822
dropout_60/PartitionedCall?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall#dropout_60/PartitionedCall:output:0conv2d_60_671298conv2d_60_671300*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_60_layer_call_and_return_conditional_losses_6710062#
!conv2d_60/StatefulPartitionedCall?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0batch_normalization_45_671303batch_normalization_45_671305batch_normalization_45_671307batch_normalization_45_671309*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_67105920
.batch_normalization_45/StatefulPartitionedCall?
dropout_61/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_61_layer_call_and_return_conditional_losses_6711122
dropout_61/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall#dropout_61/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_6711312
flatten_15/PartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_36_671314dense_36_671316*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_6711502"
 dense_36/StatefulPartitionedCall?
.batch_normalization_46/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0batch_normalization_46_671319batch_normalization_46_671321batch_normalization_46_671323batch_normalization_46_671325*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_67069220
.batch_normalization_46/StatefulPartitionedCall?
dropout_62/PartitionedCallPartitionedCall7batch_normalization_46/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_62_layer_call_and_return_conditional_losses_6712182
dropout_62/PartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_62/PartitionedCall:output:0dense_37_671329dense_37_671331*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_6712422"
 dense_37/StatefulPartitionedCall?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_44/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_46/StatefulPartitionedCall"^conv2d_57/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????::::::::::::::::::::::::::::2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_44/StatefulPartitionedCall.batch_normalization_44/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_46/StatefulPartitionedCall.batch_normalization_46/StatefulPartitionedCall2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_57_input
?

*__inference_conv2d_59_layer_call_fn_672410

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_59_layer_call_and_return_conditional_losses_6709482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
+__inference_dropout_59_layer_call_fn_672385

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_59_layer_call_and_return_conditional_losses_6709192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
F__inference_dropout_61_layer_call_and_return_conditional_losses_671112

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????		?2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????		?2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
G
+__inference_dropout_60_layer_call_fn_672437

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_60_layer_call_and_return_conditional_losses_6709822
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_672255

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_45_layer_call_fn_672508

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_6710412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????		?::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_672559

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_61_layer_call_and_return_conditional_losses_672597

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????		?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????		?*
dtype0*
seed?2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????		?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????		?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????		?2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_45_layer_call_fn_672521

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_6710592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????		?::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
conv2d_57_input@
!serving_default_conv2d_57_input:0?????????<
dense_370
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?~
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
	optimizer

signatures
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?x
_tf_keras_sequential?x{"class_name": "Sequential", "name": "sequential_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_57_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_57", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_43", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_58", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_44", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_59", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_59", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_60", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_60", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_45", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_61", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_46", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_62", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_57_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_57", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_43", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_58", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_44", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_59", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_59", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_60", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_60", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_45", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_61", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_46", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_62", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_57", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?	
axis
	 gamma
!beta
"moving_mean
#moving_variance
#$_self_saveable_object_factories
%regularization_losses
&trainable_variables
'	variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_43", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_43", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 32]}}
?

)kernel
*bias
#+_self_saveable_object_factories
,regularization_losses
-trainable_variables
.	variables
/	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_58", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_58", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 32]}}
?	
0axis
	1gamma
2beta
3moving_mean
4moving_variance
#5_self_saveable_object_factories
6regularization_losses
7trainable_variables
8	variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_44", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_44", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 32]}}
?
#:_self_saveable_object_factories
;regularization_losses
<trainable_variables
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_59", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_59", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?


?kernel
@bias
#A_self_saveable_object_factories
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_59", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_59", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 32]}}
?
#F_self_saveable_object_factories
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
#K_self_saveable_object_factories
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_60", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?


Pkernel
Qbias
#R_self_saveable_object_factories
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_60", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 64]}}
?	
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
#\_self_saveable_object_factories
]regularization_losses
^trainable_variables
_	variables
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_45", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_45", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 128]}}
?
#a_self_saveable_object_factories
bregularization_losses
ctrainable_variables
d	variables
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_61", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_61", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
#f_self_saveable_object_factories
gregularization_losses
htrainable_variables
i	variables
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kkernel
lbias
#m_self_saveable_object_factories
nregularization_losses
otrainable_variables
p	variables
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10368}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10368]}}
?	
raxis
	sgamma
tbeta
umoving_mean
vmoving_variance
#w_self_saveable_object_factories
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_46", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_46", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
#|_self_saveable_object_factories
}regularization_losses
~trainable_variables
	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_62", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_62", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
?kernel
	?bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
"
	optimizer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
0
1
 2
!3
)4
*5
16
27
?8
@9
P10
Q11
X12
Y13
k14
l15
s16
t17
?18
?19"
trackable_list_wrapper
?
0
1
 2
!3
"4
#5
)6
*7
18
29
310
411
?12
@13
P14
Q15
X16
Y17
Z18
[19
k20
l21
s22
t23
u24
v25
?26
?27"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
regularization_losses
trainable_variables
 ?layer_regularization_losses
?layers
	variables
?layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_57/kernel
: 2conv2d_57/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
regularization_losses
trainable_variables
 ?layer_regularization_losses
?layers
	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_43/gamma
):' 2batch_normalization_43/beta
2:0  (2"batch_normalization_43/moving_mean
6:4  (2&batch_normalization_43/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
<
 0
!1
"2
#3"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
%regularization_losses
&trainable_variables
 ?layer_regularization_losses
?layers
'	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_58/kernel
: 2conv2d_58/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
,regularization_losses
-trainable_variables
 ?layer_regularization_losses
?layers
.	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_44/gamma
):' 2batch_normalization_44/beta
2:0  (2"batch_normalization_44/moving_mean
6:4  (2&batch_normalization_44/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
<
10
21
32
43"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
6regularization_losses
7trainable_variables
 ?layer_regularization_losses
?layers
8	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
;regularization_losses
<trainable_variables
 ?layer_regularization_losses
?layers
=	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_59/kernel
:@2conv2d_59/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
Bregularization_losses
Ctrainable_variables
 ?layer_regularization_losses
?layers
D	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
Gregularization_losses
Htrainable_variables
 ?layer_regularization_losses
?layers
I	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
Lregularization_losses
Mtrainable_variables
 ?layer_regularization_losses
?layers
N	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)@?2conv2d_60/kernel
:?2conv2d_60/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
Sregularization_losses
Ttrainable_variables
 ?layer_regularization_losses
?layers
U	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_45/gamma
*:(?2batch_normalization_45/beta
3:1? (2"batch_normalization_45/moving_mean
7:5? (2&batch_normalization_45/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
<
X0
Y1
Z2
[3"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
]regularization_losses
^trainable_variables
 ?layer_regularization_losses
?layers
_	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
bregularization_losses
ctrainable_variables
 ?layer_regularization_losses
?layers
d	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
gregularization_losses
htrainable_variables
 ?layer_regularization_losses
?layers
i	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
?Q?2dense_36/kernel
:?2dense_36/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
nregularization_losses
otrainable_variables
 ?layer_regularization_losses
?layers
p	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_46/gamma
*:(?2batch_normalization_46/beta
3:1? (2"batch_normalization_46/moving_mean
7:5? (2&batch_normalization_46/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
<
s0
t1
u2
v3"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
xregularization_losses
ytrainable_variables
 ?layer_regularization_losses
?layers
z	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
}regularization_losses
~trainable_variables
 ?layer_regularization_losses
?layers
	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?
2dense_37/kernel
:
2dense_37/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?regularization_losses
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
X
"0
#1
32
43
Z4
[5
u6
v7"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
!__inference__wrapped_model_670239?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *6?3
1?.
conv2d_57_input?????????
?2?
I__inference_sequential_17_layer_call_and_return_conditional_losses_671945
I__inference_sequential_17_layer_call_and_return_conditional_losses_671259
I__inference_sequential_17_layer_call_and_return_conditional_losses_671335
I__inference_sequential_17_layer_call_and_return_conditional_losses_671834?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_sequential_17_layer_call_fn_671610
.__inference_sequential_17_layer_call_fn_672067
.__inference_sequential_17_layer_call_fn_671473
.__inference_sequential_17_layer_call_fn_672006?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv2d_57_layer_call_and_return_conditional_losses_672078?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_57_layer_call_fn_672087?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_672189
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_672125
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_672171
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_672107?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_43_layer_call_fn_672215
7__inference_batch_normalization_43_layer_call_fn_672151
7__inference_batch_normalization_43_layer_call_fn_672138
7__inference_batch_normalization_43_layer_call_fn_672202?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv2d_58_layer_call_and_return_conditional_losses_672226?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_58_layer_call_fn_672235?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_672255
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_672337
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_672273
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_672319?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_44_layer_call_fn_672363
7__inference_batch_normalization_44_layer_call_fn_672299
7__inference_batch_normalization_44_layer_call_fn_672350
7__inference_batch_normalization_44_layer_call_fn_672286?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_59_layer_call_and_return_conditional_losses_672380
F__inference_dropout_59_layer_call_and_return_conditional_losses_672375?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_59_layer_call_fn_672390
+__inference_dropout_59_layer_call_fn_672385?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv2d_59_layer_call_and_return_conditional_losses_672401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_59_layer_call_fn_672410?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_670453?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
1__inference_max_pooling2d_23_layer_call_fn_670459?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_dropout_60_layer_call_and_return_conditional_losses_672422
F__inference_dropout_60_layer_call_and_return_conditional_losses_672427?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_60_layer_call_fn_672437
+__inference_dropout_60_layer_call_fn_672432?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv2d_60_layer_call_and_return_conditional_losses_672448?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_60_layer_call_fn_672457?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_672559
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_672541
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_672495
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_672477?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_45_layer_call_fn_672585
7__inference_batch_normalization_45_layer_call_fn_672508
7__inference_batch_normalization_45_layer_call_fn_672521
7__inference_batch_normalization_45_layer_call_fn_672572?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_61_layer_call_and_return_conditional_losses_672602
F__inference_dropout_61_layer_call_and_return_conditional_losses_672597?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_61_layer_call_fn_672607
+__inference_dropout_61_layer_call_fn_672612?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_flatten_15_layer_call_and_return_conditional_losses_672618?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_flatten_15_layer_call_fn_672623?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_36_layer_call_and_return_conditional_losses_672634?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_36_layer_call_fn_672643?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_672679
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_672699?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_46_layer_call_fn_672725
7__inference_batch_normalization_46_layer_call_fn_672712?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_62_layer_call_and_return_conditional_losses_672742
F__inference_dropout_62_layer_call_and_return_conditional_losses_672737?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_62_layer_call_fn_672752
+__inference_dropout_62_layer_call_fn_672747?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dense_37_layer_call_and_return_conditional_losses_672763?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_37_layer_call_fn_672772?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_671673conv2d_57_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_670239? !"#)*1234?@PQXYZ[klvsut??@?=
6?3
1?.
conv2d_57_input?????????
? "3?0
.
dense_37"?
dense_37?????????
?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_672107r !"#;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_672125r !"#;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_672171? !"#M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
R__inference_batch_normalization_43_layer_call_and_return_conditional_losses_672189? !"#M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
7__inference_batch_normalization_43_layer_call_fn_672138e !"#;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
7__inference_batch_normalization_43_layer_call_fn_672151e !"#;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
7__inference_batch_normalization_43_layer_call_fn_672202? !"#M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
7__inference_batch_normalization_43_layer_call_fn_672215? !"#M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_672255?1234M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_672273?1234M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_672319r1234;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
R__inference_batch_normalization_44_layer_call_and_return_conditional_losses_672337r1234;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
7__inference_batch_normalization_44_layer_call_fn_672286?1234M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
7__inference_batch_normalization_44_layer_call_fn_672299?1234M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
7__inference_batch_normalization_44_layer_call_fn_672350e1234;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
7__inference_batch_normalization_44_layer_call_fn_672363e1234;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_672477tXYZ[<?9
2?/
)?&
inputs?????????		?
p
? ".?+
$?!
0?????????		?
? ?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_672495tXYZ[<?9
2?/
)?&
inputs?????????		?
p 
? ".?+
$?!
0?????????		?
? ?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_672541?XYZ[N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_45_layer_call_and_return_conditional_losses_672559?XYZ[N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
7__inference_batch_normalization_45_layer_call_fn_672508gXYZ[<?9
2?/
)?&
inputs?????????		?
p
? "!??????????		??
7__inference_batch_normalization_45_layer_call_fn_672521gXYZ[<?9
2?/
)?&
inputs?????????		?
p 
? "!??????????		??
7__inference_batch_normalization_45_layer_call_fn_672572?XYZ[N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
7__inference_batch_normalization_45_layer_call_fn_672585?XYZ[N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_672679duvst4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
R__inference_batch_normalization_46_layer_call_and_return_conditional_losses_672699dvsut4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
7__inference_batch_normalization_46_layer_call_fn_672712Wuvst4?1
*?'
!?
inputs??????????
p
? "????????????
7__inference_batch_normalization_46_layer_call_fn_672725Wvsut4?1
*?'
!?
inputs??????????
p 
? "????????????
E__inference_conv2d_57_layer_call_and_return_conditional_losses_672078l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_57_layer_call_fn_672087_7?4
-?*
(?%
inputs?????????
? " ?????????? ?
E__inference_conv2d_58_layer_call_and_return_conditional_losses_672226l)*7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_58_layer_call_fn_672235_)*7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
E__inference_conv2d_59_layer_call_and_return_conditional_losses_672401l?@7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_59_layer_call_fn_672410_?@7?4
-?*
(?%
inputs????????? 
? " ??????????@?
E__inference_conv2d_60_layer_call_and_return_conditional_losses_672448mPQ7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0?????????		?
? ?
*__inference_conv2d_60_layer_call_fn_672457`PQ7?4
-?*
(?%
inputs?????????@
? "!??????????		??
D__inference_dense_36_layer_call_and_return_conditional_losses_672634^kl0?-
&?#
!?
inputs??????????Q
? "&?#
?
0??????????
? ~
)__inference_dense_36_layer_call_fn_672643Qkl0?-
&?#
!?
inputs??????????Q
? "????????????
D__inference_dense_37_layer_call_and_return_conditional_losses_672763_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? 
)__inference_dense_37_layer_call_fn_672772R??0?-
&?#
!?
inputs??????????
? "??????????
?
F__inference_dropout_59_layer_call_and_return_conditional_losses_672375l;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
F__inference_dropout_59_layer_call_and_return_conditional_losses_672380l;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
+__inference_dropout_59_layer_call_fn_672385_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
+__inference_dropout_59_layer_call_fn_672390_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
F__inference_dropout_60_layer_call_and_return_conditional_losses_672422l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
F__inference_dropout_60_layer_call_and_return_conditional_losses_672427l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
+__inference_dropout_60_layer_call_fn_672432_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
+__inference_dropout_60_layer_call_fn_672437_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
F__inference_dropout_61_layer_call_and_return_conditional_losses_672597n<?9
2?/
)?&
inputs?????????		?
p
? ".?+
$?!
0?????????		?
? ?
F__inference_dropout_61_layer_call_and_return_conditional_losses_672602n<?9
2?/
)?&
inputs?????????		?
p 
? ".?+
$?!
0?????????		?
? ?
+__inference_dropout_61_layer_call_fn_672607a<?9
2?/
)?&
inputs?????????		?
p
? "!??????????		??
+__inference_dropout_61_layer_call_fn_672612a<?9
2?/
)?&
inputs?????????		?
p 
? "!??????????		??
F__inference_dropout_62_layer_call_and_return_conditional_losses_672737^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
F__inference_dropout_62_layer_call_and_return_conditional_losses_672742^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
+__inference_dropout_62_layer_call_fn_672747Q4?1
*?'
!?
inputs??????????
p
? "????????????
+__inference_dropout_62_layer_call_fn_672752Q4?1
*?'
!?
inputs??????????
p 
? "????????????
F__inference_flatten_15_layer_call_and_return_conditional_losses_672618b8?5
.?+
)?&
inputs?????????		?
? "&?#
?
0??????????Q
? ?
+__inference_flatten_15_layer_call_fn_672623U8?5
.?+
)?&
inputs?????????		?
? "???????????Q?
L__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_670453?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_23_layer_call_fn_670459?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_sequential_17_layer_call_and_return_conditional_losses_671259? !"#)*1234?@PQXYZ[kluvst??H?E
>?;
1?.
conv2d_57_input?????????
p

 
? "%?"
?
0?????????

? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_671335? !"#)*1234?@PQXYZ[klvsut??H?E
>?;
1?.
conv2d_57_input?????????
p 

 
? "%?"
?
0?????????

? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_671834? !"#)*1234?@PQXYZ[kluvst????<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????

? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_671945? !"#)*1234?@PQXYZ[klvsut????<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
.__inference_sequential_17_layer_call_fn_671473? !"#)*1234?@PQXYZ[kluvst??H?E
>?;
1?.
conv2d_57_input?????????
p

 
? "??????????
?
.__inference_sequential_17_layer_call_fn_671610? !"#)*1234?@PQXYZ[klvsut??H?E
>?;
1?.
conv2d_57_input?????????
p 

 
? "??????????
?
.__inference_sequential_17_layer_call_fn_672006{ !"#)*1234?@PQXYZ[kluvst????<
5?2
(?%
inputs?????????
p

 
? "??????????
?
.__inference_sequential_17_layer_call_fn_672067{ !"#)*1234?@PQXYZ[klvsut????<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
$__inference_signature_wrapper_671673? !"#)*1234?@PQXYZ[klvsut??S?P
? 
I?F
D
conv2d_57_input1?.
conv2d_57_input?????????"3?0
.
dense_37"?
dense_37?????????
