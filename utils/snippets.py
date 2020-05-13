def tensorboard():
    print("""
    %load_ext tensorboard
    %tensorboard --logdir logs/ --port 6006

    Now open: http://localhost:6006
    """)

def imports():
    print("""
    # Add directory to path
    import sys; sys.path.insert(0, '/home/deeplearning/code/recursos')
    """)
