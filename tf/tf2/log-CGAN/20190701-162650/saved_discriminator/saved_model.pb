фо
╢¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.0.0-dev201906032v1.12.1-3166-ge1c98eeb8f8┘Щ
z
disc_F1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:
КА*
shared_namedisc_F1/kernel
Ц
"disc_F1/kernel/Read/ReadVariableOpReadVariableOpdisc_F1/kernel*!
_class
loc:@disc_F1/kernel*
dtype0* 
_output_shapes
:
КА
q
disc_F1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:А*
shared_namedisc_F1/bias
Л
 disc_F1/bias/Read/ReadVariableOpReadVariableOpdisc_F1/bias*
_class
loc:@disc_F1/bias*
dtype0*
_output_shapes	
:А
z
disc_F2/kernelVarHandleOp*
shared_namedisc_F2/kernel*
dtype0*
_output_shapes
: *
shape:
АА
Ц
"disc_F2/kernel/Read/ReadVariableOpReadVariableOpdisc_F2/kernel*
dtype0* 
_output_shapes
:
АА*!
_class
loc:@disc_F2/kernel
q
disc_F2/biasVarHandleOp*
shared_namedisc_F2/bias*
dtype0*
_output_shapes
: *
shape:А
Л
 disc_F2/bias/Read/ReadVariableOpReadVariableOpdisc_F2/bias*
_class
loc:@disc_F2/bias*
dtype0*
_output_shapes	
:А
y
disc_F3/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	А*
shared_namedisc_F3/kernel
Х
"disc_F3/kernel/Read/ReadVariableOpReadVariableOpdisc_F3/kernel*
dtype0*
_output_shapes
:	А*!
_class
loc:@disc_F3/kernel
p
disc_F3/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_namedisc_F3/bias
К
 disc_F3/bias/Read/ReadVariableOpReadVariableOpdisc_F3/bias*
dtype0*
_output_shapes
:*
_class
loc:@disc_F3/bias

NoOpNoOp
з
ConstConst"/device:CPU:0*т
value╪B╒ B╬
╣
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6

signatures
 
 
 
 
&
	
activation


kernel
bias
&

activation

kernel
bias


kernel
bias
 
 
ZX
VARIABLE_VALUEdisc_F1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdisc_F1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
ZX
VARIABLE_VALUEdisc_F2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdisc_F2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdisc_F3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdisc_F3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
О
serving_default_disc_IN-IMGPlaceholder*$
shape:           *
dtype0*/
_output_shapes
:           
А
serving_default_disc_IN-LABELPlaceholder*
dtype0*'
_output_shapes
:         
*
shape:         

є
StatefulPartitionedCallStatefulPartitionedCallserving_default_disc_IN-IMGserving_default_disc_IN-LABELdisc_F1/kerneldisc_F1/biasdisc_F2/kerneldisc_F2/biasdisc_F3/kerneldisc_F3/bias**
config_proto

CPU

GPU 2J 8*
Tin

2*'
_output_shapes
:         *0
f+R)
'__inference_signature_wrapper_238611463*
Tout
2
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
ф
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"disc_F1/kernel/Read/ReadVariableOp disc_F1/bias/Read/ReadVariableOp"disc_F2/kernel/Read/ReadVariableOp disc_F2/bias/Read/ReadVariableOp"disc_F3/kernel/Read/ReadVariableOp disc_F3/bias/Read/ReadVariableOpConst*0
_gradient_op_typePartitionedCall-238611509*+
f&R$
"__inference__traced_save_238611508*
Tout
2**
config_proto

CPU

GPU 2J 8*
_output_shapes
: *
Tin

2
ч
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedisc_F1/kerneldisc_F1/biasdisc_F2/kerneldisc_F2/biasdisc_F3/kerneldisc_F3/bias*
Tin
	2*
_output_shapes
: *0
_gradient_op_typePartitionedCall-238611540*.
f)R'
%__inference__traced_restore_238611539*
Tout
2**
config_proto

CPU

GPU 2J 8оy
ё	
╨
'__inference_signature_wrapper_238611463
disc_in_img
disc_in_label"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCalldisc_in_imgdisc_in_labelstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin

2*0
_gradient_op_typePartitionedCall-238611454*-
f(R&
$__inference__wrapped_model_238611447*
Tout
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*Y
_input_shapesH
F:           :         
::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : :+ '
%
_user_specified_namedisc_IN-IMG:-)
'
_user_specified_namedisc_IN-LABEL
▀
Щ
"__inference__traced_save_238611508
file_prefix-
)savev2_disc_f1_kernel_read_readvariableop+
'savev2_disc_f1_bias_read_readvariableop-
)savev2_disc_f2_kernel_read_readvariableop+
'savev2_disc_f2_bias_read_readvariableop-
)savev2_disc_f3_kernel_read_readvariableop+
'savev2_disc_f3_bias_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_dc1f41eea8ac4efb887bb2236ec7a851/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ╢
SaveV2/tensor_namesConst"/device:CPU:0*▀
value╒B╥B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:y
SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B B *
dtype0*
_output_shapes
:г
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_disc_f1_kernel_read_readvariableop'savev2_disc_f1_bias_read_readvariableop)savev2_disc_f2_kernel_read_readvariableop'savev2_disc_f2_bias_read_readvariableop)savev2_disc_f3_kernel_read_readvariableop'savev2_disc_f3_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

2h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:Ц
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*N
_input_shapes=
;: :
КА:А:
АА:А:	А:: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : 
в
├
%__inference__traced_restore_238611539
file_prefix#
assignvariableop_disc_f1_kernel#
assignvariableop_1_disc_f1_bias%
!assignvariableop_2_disc_f2_kernel#
assignvariableop_3_disc_f2_bias%
!assignvariableop_4_disc_f3_kernel#
assignvariableop_5_disc_f3_bias

identity_7ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5в	RestoreV2вRestoreV2_1╣
RestoreV2/tensor_namesConst"/device:CPU:0*▀
value╒B╥B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:|
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B B *
dtype0*
_output_shapes
:╝
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:{
AssignVariableOpAssignVariableOpassignvariableop_disc_f1_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_disc_f1_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:Б
AssignVariableOp_2AssignVariableOp!assignvariableop_2_disc_f2_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0
AssignVariableOp_3AssignVariableOpassignvariableop_3_disc_f2_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:Б
AssignVariableOp_4AssignVariableOp!assignvariableop_4_disc_f3_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_disc_f3_biasIdentity_5:output:0*
dtype0*
_output_shapes
 М
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B ╡
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ╓

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: т

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_52
RestoreV2_1RestoreV2_12
	RestoreV2	RestoreV22(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:+ '
%
_user_specified_namefile_prefix: : : : : : 
░5
Н
$__inference__wrapped_model_238611447
disc_in_img
disc_in_label=
9cgan_discriminator_disc_f1_matmul_readvariableop_resource>
:cgan_discriminator_disc_f1_biasadd_readvariableop_resource=
9cgan_discriminator_disc_f2_matmul_readvariableop_resource>
:cgan_discriminator_disc_f2_biasadd_readvariableop_resource=
9cgan_discriminator_disc_f3_matmul_readvariableop_resource>
:cgan_discriminator_disc_f3_biasadd_readvariableop_resource
identityИв1CGAN-Discriminator/disc_F1/BiasAdd/ReadVariableOpв0CGAN-Discriminator/disc_F1/MatMul/ReadVariableOpв1CGAN-Discriminator/disc_F2/BiasAdd/ReadVariableOpв0CGAN-Discriminator/disc_F2/MatMul/ReadVariableOpв1CGAN-Discriminator/disc_F3/BiasAdd/ReadVariableOpв0CGAN-Discriminator/disc_F3/MatMul/ReadVariableOpb
'CGAN-Discriminator/disc_IN-IMG-1D/ShapeShapedisc_in_img*
T0*
_output_shapes
:
5CGAN-Discriminator/disc_IN-IMG-1D/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:Б
7CGAN-Discriminator/disc_IN-IMG-1D/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Б
7CGAN-Discriminator/disc_IN-IMG-1D/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:√
/CGAN-Discriminator/disc_IN-IMG-1D/strided_sliceStridedSlice0CGAN-Discriminator/disc_IN-IMG-1D/Shape:output:0>CGAN-Discriminator/disc_IN-IMG-1D/strided_slice/stack:output:0@CGAN-Discriminator/disc_IN-IMG-1D/strided_slice/stack_1:output:0@CGAN-Discriminator/disc_IN-IMG-1D/strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask|
1CGAN-Discriminator/disc_IN-IMG-1D/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
         █
/CGAN-Discriminator/disc_IN-IMG-1D/Reshape/shapePack8CGAN-Discriminator/disc_IN-IMG-1D/strided_slice:output:0:CGAN-Discriminator/disc_IN-IMG-1D/Reshape/shape/1:output:0*
T0*
N*
_output_shapes
:о
)CGAN-Discriminator/disc_IN-IMG-1D/ReshapeReshapedisc_in_img8CGAN-Discriminator/disc_IN-IMG-1D/Reshape/shape:output:0*
T0*(
_output_shapes
:         Аl
*CGAN-Discriminator/disc_IN-ALL/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: х
%CGAN-Discriminator/disc_IN-ALL/concatConcatV22CGAN-Discriminator/disc_IN-IMG-1D/Reshape:output:0disc_in_label3CGAN-Discriminator/disc_IN-ALL/concat/axis:output:0*
T0*
N*(
_output_shapes
:         К┌
0CGAN-Discriminator/disc_F1/MatMul/ReadVariableOpReadVariableOp9cgan_discriminator_disc_f1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
КА╚
!CGAN-Discriminator/disc_F1/MatMulMatMul.CGAN-Discriminator/disc_IN-ALL/concat:output:08CGAN-Discriminator/disc_F1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0╫
1CGAN-Discriminator/disc_F1/BiasAdd/ReadVariableOpReadVariableOp:cgan_discriminator_disc_f1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:А╚
"CGAN-Discriminator/disc_F1/BiasAddBiasAdd+CGAN-Discriminator/disc_F1/MatMul:product:09CGAN-Discriminator/disc_F1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЦ
2CGAN-Discriminator/disc_F1/leaky_re_lu_3/LeakyRelu	LeakyRelu+CGAN-Discriminator/disc_F1/BiasAdd:output:0*(
_output_shapes
:         А┌
0CGAN-Discriminator/disc_F2/MatMul/ReadVariableOpReadVariableOp9cgan_discriminator_disc_f2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
АА┌
!CGAN-Discriminator/disc_F2/MatMulMatMul@CGAN-Discriminator/disc_F1/leaky_re_lu_3/LeakyRelu:activations:08CGAN-Discriminator/disc_F2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╫
1CGAN-Discriminator/disc_F2/BiasAdd/ReadVariableOpReadVariableOp:cgan_discriminator_disc_f2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:А╚
"CGAN-Discriminator/disc_F2/BiasAddBiasAdd+CGAN-Discriminator/disc_F2/MatMul:product:09CGAN-Discriminator/disc_F2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЦ
2CGAN-Discriminator/disc_F2/leaky_re_lu_4/LeakyRelu	LeakyRelu+CGAN-Discriminator/disc_F2/BiasAdd:output:0*(
_output_shapes
:         А┘
0CGAN-Discriminator/disc_F3/MatMul/ReadVariableOpReadVariableOp9cgan_discriminator_disc_f3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А┘
!CGAN-Discriminator/disc_F3/MatMulMatMul@CGAN-Discriminator/disc_F2/leaky_re_lu_4/LeakyRelu:activations:08CGAN-Discriminator/disc_F3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ╓
1CGAN-Discriminator/disc_F3/BiasAdd/ReadVariableOpReadVariableOp:cgan_discriminator_disc_f3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:╟
"CGAN-Discriminator/disc_F3/BiasAddBiasAdd+CGAN-Discriminator/disc_F3/MatMul:product:09CGAN-Discriminator/disc_F3/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0М
"CGAN-Discriminator/disc_F3/SigmoidSigmoid+CGAN-Discriminator/disc_F3/BiasAdd:output:0*
T0*'
_output_shapes
:         г
IdentityIdentity&CGAN-Discriminator/disc_F3/Sigmoid:y:02^CGAN-Discriminator/disc_F1/BiasAdd/ReadVariableOp1^CGAN-Discriminator/disc_F1/MatMul/ReadVariableOp2^CGAN-Discriminator/disc_F2/BiasAdd/ReadVariableOp1^CGAN-Discriminator/disc_F2/MatMul/ReadVariableOp2^CGAN-Discriminator/disc_F3/BiasAdd/ReadVariableOp1^CGAN-Discriminator/disc_F3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*Y
_input_shapesH
F:           :         
::::::2d
0CGAN-Discriminator/disc_F3/MatMul/ReadVariableOp0CGAN-Discriminator/disc_F3/MatMul/ReadVariableOp2f
1CGAN-Discriminator/disc_F3/BiasAdd/ReadVariableOp1CGAN-Discriminator/disc_F3/BiasAdd/ReadVariableOp2f
1CGAN-Discriminator/disc_F2/BiasAdd/ReadVariableOp1CGAN-Discriminator/disc_F2/BiasAdd/ReadVariableOp2d
0CGAN-Discriminator/disc_F2/MatMul/ReadVariableOp0CGAN-Discriminator/disc_F2/MatMul/ReadVariableOp2f
1CGAN-Discriminator/disc_F1/BiasAdd/ReadVariableOp1CGAN-Discriminator/disc_F1/BiasAdd/ReadVariableOp2d
0CGAN-Discriminator/disc_F1/MatMul/ReadVariableOp0CGAN-Discriminator/disc_F1/MatMul/ReadVariableOp: : : : : : :+ '
%
_user_specified_namedisc_IN-IMG:-)
'
_user_specified_namedisc_IN-LABEL"6L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*Г
serving_defaultя
G
disc_IN-LABEL6
serving_default_disc_IN-LABEL:0         

K
disc_IN-IMG<
serving_default_disc_IN-IMG:0           ;
disc_F30
StatefulPartitionedCall:0         tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:Ц
Ї
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6

signatures
_default_save_signature"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
D
	
activation


kernel
bias"
_generic_user_object
D

activation

kernel
bias"
_generic_user_object
4

kernel
bias"
_generic_user_object
,
serving_default"
signature_map
"
_generic_user_object
": 
КА2disc_F1/kernel
:А2disc_F1/bias
"
_generic_user_object
": 
АА2disc_F2/kernel
:А2disc_F2/bias
!:	А2disc_F3/kernel
:2disc_F3/bias
Ь2Щ
$__inference__wrapped_model_238611447Ё
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *`в]
[ЪX
-К*
disc_IN-IMG           
'К$
disc_IN-LABEL         

GBE
'__inference_signature_wrapper_238611463disc_IN-IMGdisc_IN-LABELЁ
'__inference_signature_wrapper_238611463─
ЖвВ
в 
{кx
8
disc_IN-LABEL'К$
disc_IN-LABEL         

<
disc_IN-IMG-К*
disc_IN-IMG           "1к.
,
disc_F3!К
disc_F3         ╨
$__inference__wrapped_model_238611447з
jвg
`в]
[ЪX
-К*
disc_IN-IMG           
'К$
disc_IN-LABEL         

к "1к.
,
disc_F3!К
disc_F3         