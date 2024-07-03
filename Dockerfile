FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade -r /code/requirements.txt

ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
RUN pip install flash-attn --no-build-isolation

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]