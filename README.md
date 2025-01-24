# whisper-test

## 启动
```
poetry install

poetry run python main.py
```

## 测试结论

whisper本身不适合实时转写，但也有解决方案（中文效果待验证）https://github.com/ufal/whisper_streaming?tab=readme-ov-file
turbo综合性能最好，适合记录转写。
如果实现一个会后生成会议纪要，接受生成的时长。可以通过whisper去转写内容，然后通过本地大模型做优化。

记录链接：
https://www.yuque.com/douqiting/nnikg7/gq6lofc8oyxt5lfa