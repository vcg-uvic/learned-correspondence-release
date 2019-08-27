
# Learning to Find Good Correspondences (CVPR 2018)

This repository is a reference implementation for K. Yi\*, E. Trulls\*, Y. Ono,
V. Lepetit, M. Salzmann, and P. Fua, "Learning to Find Good Correspondences",
CVPR 2018 (* equal contributions). If you use this code in your research,
please cite the paper.

# Installation

This code base is based on Python3. For more details on the required libraries,
see `requirements.txt`. You can also easily prepare this by doing

```
pip install -r requirements.txt
```
# Preparing data

Download the
[brown_bm](http://webhome.cs.uvic.ca/~kyi/files/2018/learned-correspondence/brown_bm_3---brown_bm_3-maxpairs-10000-random---skip-10-dilate-25.tar.gz)
sequence and
[st_peters](http://webhome.cs.uvic.ca/~kyi/files/2018/learned-correspondence/st_peters_square.tar.gz)
sequence and extract them in the datasets directory. For example, for
`st_peters` you should have a directory looking like
`./datasets/st_peters_square/train`. Also download
[reichstag](http://webhome.cs.uvic.ca/~kyi/files/2018/learned-correspondence/reichstag.tar.gz)
dataset for testing, which is quite small!

Once the datasets are downloaded, run `dump_data.py` to prepare the
datasets. The following commands should be run.

```
./dump_data.py --data_tr=st_peters --data_va=st_peters --data_te=st_peters
./dump_data.py --data_tr=brown_bm_3_05 --data_va=brown_bm_3_05 --data_te=brown_bm_3_05
./dump_data.py --data_tr=reichstag --data_va=reichstag --data_te=reichstag
```

Hang on tight, this would take a while.


# Training

While we also provide our trained models, you can also easily train your own
models. Simply run:

```
./main.py --run_mode=train
```

See `config.py` for more options in running the software. Try it
yourself. Nearly all parameters that we changed in the paper should be
there. 

For designating datasets, modify `data_tr`, `data_va`, and `data_te` to your
liking. Also you can simply do something like `st_peters.brown_bm_3_05` to
train with the combined datasets.

The default place to store the results is `./logs`. To change this, use
`res_dir` to set the base directory, `log_dir` for the suffix for the training
configurations. `test_log_dir` is used to if you want to change the suffix for
storing results. For example, `log_dir` can store the training configuration,
and `test_log_dir` can store which training configuration is used on which
testing dataset.

# Testing

Again, testing is quite simple. After training is done, run:

```
./main.py --run_mode=test

```

Or if you simply want to test the pretrained model on your dataset, you can:

```
./main.py --run_mode=test --res_dir="./" --log_dir="models" --test_log_dir="results" --data_tr="st_peters.brown_bm_3_05" --data_va="reichstag" --data_te="reichstag" 

# Or use following command to save an additional numpy structured output (Thanks to GrumpyZhou)
./main.py --run_mode=test_simple --res_dir="./" --log_dir="models" --test_log_dir="results"  --data_va="reichstag" --data_te="reichstag"
```

This time, it should only take about 30 seconds since `reichstag` is tiny, but
depending on the dataset, this might take a while (roughly seven minutes on
`st_peters`). This is because you are testing on the entire test dataset. One
thing to note is that, as written on the paper, most the computation is done in
the CPU, and GPU is not really necessary for testing.

Also, as of now, there is no stand-alone testing code for a single image
pair. The current code-base also evaluates testing on the validation dataset as
well, since the only validation performed while training is with the
eight-point algorithm. We welcome contributions for untying this mess we have.

Once run, you'll find a TensorBoard log files and some txt files that contain
the results of the expriments.


# Notes on implementation

For the dataset generation, we used OpenCV SIFT with fast-math flag on. We've
noticed that when using `opencv-contrib-python` package from pip, you get
different results, slightly different from the paper.



