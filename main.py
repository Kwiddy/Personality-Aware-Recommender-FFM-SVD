from reviewAPR import review_APR
from dataLoader import getDF

# ADD LOADING THE DATA INTO A PANDAS DATAFRAME
file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Movies_and_TV_5.json.gz'
full_df = getDF(file_path)
print(full_df)

# FOR EACH ROW IN DATA, INPUT ONLY THE USER_ID AND THE REVIEW
review_APR()