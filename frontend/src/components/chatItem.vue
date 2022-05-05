<!--聊天框中的message组件-->
<template>
  <div class="record-wrapper">
    <div class="time">{{ displayedTime }}</div>

    <div v-if="from == 1" class="msg msg-right">
      <div class="img-wrapper">
        <a-avatar class="img">USER</a-avatar>
      </div>
      <div class="message-wrapper message-wrapper-right">
        <div class="message">{{ message }}</div>
      </div>
    </div>

    <div v-else class="msg msg-left">
        <div class="img-wrapper">
          <img class="img" :src="from2"/>
        </div>

      <div class="message-wrapper message-wrapper-left">
        <div class="message" v-html="message"></div>
      </div>

    </div>

  </div>
</template>

<script>
  export default {
    name: "chatItem",
    props: {
      from: Number, // 1: 自己    2: 别人
      message: String,
      timestamp: Date,
      displayedTime: String,
      questionType: Number
    },
    data() {
      return {
        from2: require("../assets/service.png"),
      };
    },
    mounted() {
    }
  };
</script>

<style lang="scss" scoped>
  .record-wrapper {
    user-select: text;
    margin: 4px;
    padding: 4px;
  }

  .time {
    text-align: center;
    font-size: smaller;
    margin: .5em;
  }

  .msg {
    display: flex;
    flex-direction: row;

    .message-wrapper {
      max-width: 420px;
      /*display: block;*/

      margin: 0px 10px 0px 10px;

      .message {
        margin: 6px 10px 6px 10px;
        font-size: 15px;
        word-wrap: break-word; //英文换行
      }
    }

    .message-wrapper-left {
      margin-top: 10px;
      border-radius: 0px 12px 12px 12px;
      color: white;
      background-color: #a51f1f;
    }

    .message-wrapper-right {
      margin-top: 10px;
      color: white;
      background-color: black;
      border-radius: 12px 0px 12px 12px;
    }

    .img {
      flex: auto;
      height: 36px;
      width: 36px;
      border-radius: 18px;
      padding-top: 2px;
      /*border: solid white 2px;*/
    }
  }

  .msg-right {
    flex-direction: row-reverse;
  }

  .msg-left {
    flex-direction: row;
  }
</style>
