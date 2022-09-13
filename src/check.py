# --------------------------------------example 1-----------------------------------------------------
class Point:
    def __init__(self, initX, initY):
        self.x = initX
        self.y = initY

    def getX(self):
        return self.x
    
    def getY(self):
        return self.y

    def distanceFromOrigin(self):
        return (self.x**2+self.y**2)**0.5

    def halfway(self, target):
        midx = (self.x + target.x) / 2
        midy = (self.y + target.y) / 2
        return Point(midx,midy)

    def __str__(self):
        # return "Tọa độ là x = {} và y = {}".format(self.x, self.y)
        return f"Tọa độ của trung điểm là x = {self.x} và y = {self.y}"
p = Point(3,4)
q = Point(5,12)
# print("Tọa độ điểm p là ",p.distanceFromOrigin())
# print(p.halfway(q))

# ---------------------------------------------------------------example 2-----------------------------------------------
class Animal:
    type = "**"
    def __init__(self,name,age):
        self.name =  name
        self.age = age
        self.hunger = 0

    def feed(self):
        self.hunger += 1

class Cat(Animal):
	type='Cat'

	def __init__(self, name, age):
		super().__init__(name, age)

class Dog(Animal):
	type='Dog'

	def __init__(self, name, age):
		super().__init__(name, age)

class Bird(Animal):
	type='Bird'

	def fly(self):
		return 'Flying ...'

	def __init__(self, name, age):
		super().__init__(name, age)

	def __str__(self):
		return f'Hi, i\'m Bird'

b = Bird("chim bồ câu",10)
print(b.fly())

