




import gin
gin.parse_config_file("config/config_gin.gin")
"""

myfun.a1 = [1,2,3]
myfun.a3 = True



"""

@@gin.configurable
def myfun(a1=1,a2=2):
   return a1,a2
  
  
  
  



