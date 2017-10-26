FROM python:latest

# update
RUN apt-get update

# install matplotlib
RUN pip3 install matplotlib notebook

# hdf5-tools
RUN apt-get install -y sudo hdf5-tools libzmq3

# install julia 0.6.0
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/0.6/julia-0.6.0-linux-x86_64.tar.gz && \
    tar -xzf julia-0.6.0-linux-x86_64.tar.gz && \
    ln -s /julia-903644385b/bin/julia /usr/local/bin/julia

# Set the working directory to /work
WORKDIR /work

# add julia packages
ADD add_packages.jl /work
RUN julia add_packages.jl

# Make port 8888 available to the world outside this container
EXPOSE 8888

# command
CMD jupyter notebook --allow-root --port=8888 --ip=0.0.0.0

###########
# usage:
# $docker run -p 8888:8888 bayesbook