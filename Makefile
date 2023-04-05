NOTEBOOKS := \
	cnn.ipynb \
	log_reg.ipynb \
	mini_batch_kmeans.ipynb \
	mobilenet.ipynb \
	random_forest.ipynb \
	visualizations.ipynb

# Define any variables needed for the notebooks
VARIABLES := \
	--code=Code

# Define the rule to execute all the notebooks
all: $(NOTEBOOKS)

# Define a rule to execute each notebook
$(NOTEBOOKS):
	jupyter nbconvert --to notebook --execute --inplace $@ 
$(VARIABLES)
