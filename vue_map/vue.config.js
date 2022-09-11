const { defineConfig } = require('@vue/cli-service')


module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    proxy: {
      '/admin': {
        //代理api
        target: 'http://localhost:6001', // 代理接口(注意只要域名就够了)
        changeOrigin: true, //是否跨域
        ws: true, // proxy websockets
        pathRewrite: {
          //重写路径
          '^/admin': '/api' //代理路径
        }
      }
    }
  }
})

