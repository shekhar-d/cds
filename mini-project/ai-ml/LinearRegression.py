class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self) -> str:
        return 'fits the given x and y with minimum error between the actual and predicted'
    def avg(self,list1):
        sum=0
        for item in list1:
            sum+=item
        return sum/len(list1)
    def var(self,list1):
        k=self.avg(list1)
        sum =0
        for item in list1:
            sum+= (item - k)**2
        return sum/len(list1)
    def cov(self,list1,list2):
        if len(list1)!= len(list2):
            return('X and y should have same length')
        else:
            xavg=self.avg(list1)
            yavg=self.avg(list2)
            sum=0
            for i in range(len(list1)):
                sum+= (list1[i]-xavg)*(list2[i]-yavg)
            return sum/(len(list1))
    def fit(self,list1,list2):
        if len(list1)!= len(list2):
            return('X and y should have same length')
        else:
            b1=self.cov(list1,list2)/self.var(list1)
            b0=self.avg(list2)-b1*self.avg(list1)
        return b1,b0
    def predict(self,list1,list2):
        b1,b0 = self.fit(list1,list2)
        y_pred = (np.ones(len(list1))*b0) + (b1*list1)
        return y_pred
    def RMSE(self,x,y):
        y_pred=self.predict(x,y)
        sum=0
        for i in range(len(y_pred)):
            sum += ((y_pred[i]-y[i])**2)
        return (sum/len(y_pred))**0.5
