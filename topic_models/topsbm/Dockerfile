FROM ubuntu:cosmic-20181018
LABEL maintainer="error.derivative@gmail.com"

ENV DEBIAN_FRONTEND noninteractive
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE DontWarn
ENV USER_HOME=/home/host/

RUN apt-get update && \
    apt-get install -y locales=2.28-0ubuntu1 \
                       gnupg=2.2.8-3ubuntu1 \
                       gnupg2=2.2.8-3ubuntu1 \
                       apt-utils=1.7.0 \
                       git=1:2.19.1-1ubuntu1 && \
    rm -rf /var/lib/apt/lists/* && \
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8

RUN echo "deb http://downloads.skewed.de/apt/cosmic cosmic universe" >> /etc/apt/sources.list && \
    echo "deb-src http://downloads.skewed.de/apt/cosmic cosmic universe" >> /etc/apt/sources.list

RUN apt-key adv --keyserver pgp.skewed.de --recv-key 612DEFB798507F25
RUN apt-get update && \
    apt-get install -y libfreetype6-dev=2.8.1-2ubuntu2 \
                    libfreetype6=2.8.1-2ubuntu2 \
                    python3-pip=9.0.1-2.3 \
                    python3-pandas=0.23.3-1fakesync1ubuntu1 \
                    python3-graph-tool=2.27-4 && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install -U nltk==3.3 \
                    pymorphy2==0.8 \
                    cairosvg==2.2.1 \
                    lxml==4.2.5 \
                    pygal==2.4.0 \
                    jupyter==1.0.0

RUN groupadd "machine" && \
    useradd -m -g machine host && \
    mkdir -p ${USER_HOME} && \
    chown host:machine ${USER_HOME}
    
USER host
WORKDIR ${USER_HOME}

COPY --chown=host:machine news_lenta.csv TopSBM.ipynb ./

RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')" \
    && git clone https://github.com/martingerlach/hSBM_Topicmodel.git
ENTRYPOINT jupyter-notebook --ip 0.0.0.0 --no-browser
