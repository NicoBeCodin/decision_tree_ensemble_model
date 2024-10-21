# 根目录下的Makefile

# 定义子目录
SUBDIRS = data_clean

# 默认目标，递归调用所有子目录的Makefile
all:
	@for dir in $(SUBDIRS); do \
	    $(MAKE) -C $$dir; \
	done

# 清理目标，递归调用子目录的clean
clean:
	@for dir in $(SUBDIRS); do \
	    $(MAKE) -C $$dir clean; \
	done
