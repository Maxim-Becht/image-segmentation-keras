	��ǘ�aZ@��ǘ�aZ@!��ǘ�aZ@      ��!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'��ǘ�aZ@���kz@1AaP��Y@Iy���x�?r0*	4333�m�@2f
/Iterator::Root::Prefetch::FlatMap[0]::Generator.���1�A@!����2�X@).���1�A@1����2�X@:Preprocessing2E
Iterator::Root���&�?!��� �Ҫ?)�0�*�?1.t�g��?:Preprocessing2O
Iterator::Root::Prefetch�J�4�?!{����?)�J�4�?1{����?:Preprocessing2X
!Iterator::Root::Prefetch::FlatMap�%��A@! ����X@){�G�zd?1�\�~�|?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI Z� @Q^*~���W@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���kz@���kz@!���kz@      ��!       "	AaP��Y@AaP��Y@!AaP��Y@*      ��!       2      ��!       :	y���x�?y���x�?!y���x�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q Z� @y^*~���W@