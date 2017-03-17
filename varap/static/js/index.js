video_manager = new Object()
video_manager.cur = 0
video_manager.list = null
video_manager.class_name = "null"
video_manager.next = function() {
  if (this.list != null) {
    if (this.cur < this.list.length - 1) {
      this.cur ++
    }
    return this.list[this.cur]
  }
  return 'invalid'
}
video_manager.prev = function() {
  if (this.list != null) {
    if (this.cur > 0) {
      this.cur --
      return this.list[this.cur]
    }
    return this.list[this.cur]
  }
  return 'invalid'
}
video_manager.curr = function() {
  if (this.list != null) {
    if (this.cur >= 0 && this.cur < this.list.length) {
      return this.list[this.cur]
    }
  }
  return 'invalid' 
}
video_manager.reset = function() {
  this.cur = 0
  this.list = null
  this.class_name = "null"
}
video_manager.cur_vidoe_name = function() {
  fields = this.curr().split('/')
  return fields[fields.length-1]
}

$('.explore_button').click(function() {
  dataset = "hmdb51"
  class_id = $(this).attr('class_id')
  class_name = $(this).attr('class_name')
  $.getJSON('/request_video_list/' + dataset + '/' + class_id, function(jd) {
    video_manager.reset()
    video_manager.list = jd
    console.log(class_name)
    video_manager.class_name = class_name
    $('#explore_title').text(video_manager.class_name)
    $('#explore_video_name').text(video_manager.cur_vidoe_name())
    $('#video_display').attr('src', video_manager.curr())
    $("#model_explore").modal()
  })
});

$('#prev_video').click(function() {
  $('#video_display').attr('src', video_manager.prev())
  $('#explore_video_name').text(video_manager.cur_vidoe_name())
});

$('#next_video').click(function() {
  $('#video_display').attr('src', video_manager.next())
  $('#explore_video_name').text(video_manager.cur_vidoe_name())
});