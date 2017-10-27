# DockerからJupyter notebookを実行する

JuliaやPythonの実行環境構築が煩わしい場合は、Dockerを使ってデモスクリプトをJupyter notebook上で動作させることができます。
Dockerのインストールに関しては公式サイトを参考ください。
* https://docs.docker.com/engine/installation/

`Dockerfile`の置いてあるディレクトリで、イメージを作成・実行します。

    $ docker build -t bayesbook .
    $ docker run -p 8888:8888 bayesbook

