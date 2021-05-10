# make and run(production)  

make ;.\ntt.exe

# make and debug  

make ;compute-sanitizer .\ntt.exe

# clean up *.o on windows

make cleanw



# 注意
- 无关联的多线程执行，要用idx确切控制线程数量
- toomany resource ： registers爆表，减少线程数量，可能是由于给每个kernel传入太多内容导致的
- 注意区分 tid和idx
- 指针和多重指针
