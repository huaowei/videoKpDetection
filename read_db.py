import pymysql


class DB():
    # cursorclass将返回结果呈字典显示 ，如果传了就应用，不传就为None
    def __init__(self, host, user, password, database, cursorclass=None):
        self.host = host  # ip地址，本机就传localhost就可以
        self.user = user  # 账户名，如果你没有更改过，那就是root
        self.password = password  # 数据库密码
        self.databases = database  # 你具体要连接的哪一个数据库
        self.cursorclass = cursorclass  # 返回结果呈字典形式  {}

    def connect(self):  # 创建连接数据库方法
        # 创建数据库连接
        self.db = pymysql.connect(host=self.host, user=self.user, password=
        self.password, database=self.databases, cursorclass=self.cursorclass)
        # 创建游标
        self.cursor = self.db.cursor()

    def close(self):  # 关闭方法
        self.cursor.close()  # 关闭游标
        self.db.close()  # 关闭数据库

    def get_one(self, sql):  # 返回一条符合条件的查询结果
        result = 0
        try:
            self.connect()
            self.cursor.execute(sql)
            result = self.cursor.fetchone()
            self.close()
        except Exception as e:
            print('select error', e)
        return result

    def get_all(self, sql):  # 返回全部符合条件的查询结果
        result = 0
        try:
            self.connect()
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            self.close()
        except Exception as e:
            print("select error", e)
        return result

    def get_one_2(self, query, params):
        self.connect()
        self.cursor.execute(query, params)
        result = self.cursor.fetchone()
        self.close()
        return result

    def get_all_2(self, query, params):
        self.connect()
        self.cursor.execute(query, params)
        result = self.cursor.fetchall()
        self.close()
        return result

    def __edit(self, sql):  # 创建主函数
        result = 1  # 设置结果集，用于调用的时候做判断
        try:  # 这里是使用的try语句来尝试进行操作
            self.connect()
            self.cursor.execute(sql)
            self.db.commit()  # 注意的是，如果是对数据库做了修改、删除、增加的操作，那么一定要commit提交，查询和创建表不需要提交
            self.close()
        except Exception as e:  # 如果操作失败，报出操作异常，且游标进行回滚
            print('error :', e)
            result = 0
            self.db.rollback()
        return result

    def insert(self, sql):
        # 插入语句  ，以下三个都是一样的，只是调用的时候，我们看起来更加清晰而已
        return self.__edit(sql)  # 通过主函数的处理，来去执行sql语句

    def delete(self, sql):  # 删除语句
        return self.__edit(sql)

    def update(self, sql):  # 修改语句
        return self.__edit(sql)
