<template>
  <div id="root-question">

    <div class="leftSide">
      <div class="scroll-wrapper" id="py">
        <chat-item
          v-for="item in chatMessages"
          :key="item.key"
          :type="item.type"
          :message="item.message"
          :timestamp="item.timestamp"
          :displayedTime="item.displayedTime"
          :from="item.from"
          :userName="item.name"
        ></chat-item>
        <a-spin :spinning="spinning">
        </a-spin>
      </div>
      <a-input-search id="sender"
                      placeholder="请输入案件基本情况..."
                      size="default"
                      @search="asking"
                      v-model="content"
                      style="margin: 1em;width: 92%;">
        <a-button style="border-top-right-radius: 15px; border-bottom-right-radius: 15px;height: 5vh;"
                  slot="enterButton" type="primary">Send
        </a-button>
      </a-input-search>
      <div id="menu">
        <a-radio-group v-model="qType" @change="select" button-style="solid">
          <a-radio-button value="1" style="width: 16vw;">纠纷类型判断</a-radio-button>
          <a-radio-button value="2" style="width: 16vw;">相关法条推荐</a-radio-button>
          <a-radio-button value="3" style="width: 16vw;">判决结果预测</a-radio-button>
          <a-radio-button value="4" style="width: 16vw;">相关文书推荐</a-radio-button>
        </a-radio-group>
      </div>
    </div>

  </div>
</template>

<script>
  const ChatItem = () => import("./chatItem.vue");
  import {mapActions, mapGetters,mapMutations} from 'vuex'

  const qType = [" 纠纷类型判断 ", " 相关法条推荐 ", " 判决结果预测 ", " 相关文书推荐 "];


  export default {
    name: "question",


    data() {
      return {
        name: "USER",
        spinning: false,
        lastTime: {},
        content: '',
        qType: 0,

      };
    },
    components: {
      "chat-item": ChatItem
    },
    computed: {
      ...mapGetters([
        'chatMessages'
      ])
    },
    async mounted() {
      let box = document.getElementsByClassName('scroll-wrapper')[0];
      box.scrollTop = box.scrollHeight;
      this.lastTime = this.chatMessages[this.chatMessages.length - 1].timestamp;
    },

    updated() {
      let box = document.getElementsByClassName('scroll-wrapper')[0];
      box.scrollTop = box.scrollHeight;
    },

    methods: {
      ...mapActions([
        'question'
      ]),
      ...mapMutations([
        'set_qType'
      ]),
      select(e) {
        this.qType = e.target.value;
        this.set_qType(e.target.value);
        this.chatMessages.push({
          type: 1,
          message: "您选择了" + qType[e.target.value - 1] + "的问题类型，请输入案件基本情况",
          key: this.chatMessages.length,
          from: 2,
          timestamp: new Date(),
          name: this.name
        });
      },
      async asking(e) {
        let time = (parseInt(new Date() - this.lastTime) / 1000 / 60) <= 3
          ? '' : new Date().toLocaleDateString() + ' ' + new Date().toLocaleTimeString();
        if (e !== '') {
          // 消息间隔超过3分钟，则显示时间
          this.lastTime = new Date();
          this.chatMessages.push({
            type: 1,
            message: e,
            key: this.chatMessages.length,
            from: 1,
            timestamp: new Date(),
            displayedTime: time,
          });
        }
        this.spinning = true;
        this.content = '';

        let answer;
        if(this.qType!=0){
          answer = await this.question(e);
        }
        else{
          answer = "抱歉，您还未选择问题类型，请先在输入框下方选择。"
        }
        answer = answer.replaceAll('\n','<br/>');
        // answer.answer = answer.answer.replaceAll('null','');
        this.chatMessages.push({
          type: 1,
          message: answer,
          key: this.chatMessages.length,
          from: 2,
          timestamp: new Date(),
          displayedTime: time,
        });
        this.spinning = false;
      },
    },
  };
</script>

<style lang="scss">
  #menu {
    padding-top: 2vh;
    padding-bottom: 3vh;
    width: 70vw;
  }

  #root-question {
    text-align: center;
    padding-top: 5vh;
    padding-left: 15vw;
    //background-size: cover;
    width: 100%;
    height: 100%;
  }

  .ant-input-group-addon {
    background-color: transparent;
  }

  #sender.ant-input {
    border-top-left-radius: 15px;
    border-bottom-left-radius: 15px;
    height: 5vh;
  }

  .scroll-wrapper {
    border-radius: 20px;
    padding: 15px;
    overflow: auto; // 非常关键
    height: 70vh;
  }

  .leftSide {
    /*border-radius: 30px;*/
    background-color: rgba(245, 248, 252, 0.95);

    /*border: solid black 2px;*/
    height: 90vh;
    width: 70vw;
    box-shadow: 0 2px 2px rgba(0, 0, 0, .2);
  }

</style>
