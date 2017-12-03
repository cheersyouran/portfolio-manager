import sys

project_path = '/Users/wangchengming/Documents/5001Project/Snowball/RL'
start_date = '2017-01-06'

sys.path.append(project_path)

if __name__ == '__main__':
	from task1.Similarity_Search.SimiSearch import run_simi_search
	run_simi_search()

