
source := fast_dep
binary: $(source).cpp
	g++ $(source).cpp -O3 -o $(source).o 

tag :=
lang := english
T := 10
threshold := 0.8
pid := $(shell echo $$$$)
dir := run_$(tag)_$(pid)
run : $(source).o $(dir)/config
	cp $(source).cpp $(dir)
	cp $(source).o $(dir)
	cp *$(lang)* $(dir)
	cp rules $(dir)
	cp dir_rules $(dir)
	chmod -R 777 $(dir)
	cd $(dir); ./$(source).o >> log.out
$(dir)/config :
	mkdir $(dir)
	sed -e 's/@lang@/$(lang)/g' config | \
	sed -e 's/@T@/$(T)/g' | \
	sed -e 's/@threshold@/$(threshold)/g' > $@



rules := 1 2 3 4 5 6 7 8 9 10 11 12 13
run_ablation : $(rules:%=run_ablation_%)

run_ablation_% : rules_%
	mkdir $@_$(tag)_$(pid)
	sed -e 's/@lang@/$(lang)/g' config | \
	sed -e 's/@T@/$(T)/g' | \
	sed -e 's/@threshold@/$(threshold)/g' > $@_$(tag)_$(pid)/config
	cp $< $@_$(tag)_$(pid)/rules
	cp $(source).cpp $@_$(tag)_$(pid)
	cp $(source).o $@_$(tag)_$(pid)
	cp *$(lang)* $@_$(tag)_$(pid)
	cp dir_rules $@_$(tag)_$(pid)
	cd $@_$(tag)_$(pid); ./$(source).o >> log.out	

rules_1 :
	sed -e '1,1d' rules > rules_1
rules_2 :
	sed -e '2,2d' rules > rules_2
rules_3:
	sed -e '3,3d' rules > rules_3
rules_4:
	sed -e '4,4d' rules > rules_4
rules_5:
	sed -e '5,5d' rules > rules_5
rules_6:
	sed -e '6,6d' rules > rules_6
rules_7:
	sed -e '7,7d' rules > rules_7
rules_8:
	sed -e '8,8d' rules > rules_8
rules_9:
	sed -e '9,9d' rules > rules_9
rules_10:
	sed -e '10,10d' rules > rules_10
rules_11:
	sed -e '11,11d' rules > rules_11
rules_12:
	sed -e '12,12d' rules > rules_12
rules_13:
	sed -e '13,13d' rules > rules_13


clean:
	rm -rf ./fast_dep.o || true
	rm -rf ./run__* || true