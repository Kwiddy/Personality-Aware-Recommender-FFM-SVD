from reviewAPR import review_APR
from dataLoader import getDF
from svd import create_svd
from datetime import date, datetime

# track runtime 
start = datetime.now()

# ADD LOADING THE DATA INTO A PANDAS DATAFRAME
file_path = './Datasets/jianmoNI_UCSD_Amazon_Review_Data/2018/small/5-core/Movies_and_TV_5.json.gz'

new_path = [char for char in file_path]
i = -1
while new_path[i] != "/":
    del new_path[i]
parent_path = ''.join(new_path)
new_path = parent_path + "Movie_and_TV_5.csv"

full_df= getDF(file_path, parent_path)

# FOR EACH ROW IN DATA, INPUT ONLY THE USER_ID AND THE REVIEW
ffm_df = review_APR(full_df, parent_path)

chosen_user = "A2M1CU2IRZG0K9"

# choose method
print("")
print("Methods:")
print("[S] - SVD")
valid_in = False
while not valid_in:
    method = input("Please choose a method above: ")
    if method.upper() == "S":
        valid_in = True
        create_svd(full_df, ffm_df, chosen_user)



print("Runtime: ", datetime.now()-start)