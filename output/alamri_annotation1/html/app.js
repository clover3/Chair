var annotApp = angular.module('annotApp', []);


(function () {
    'use strict';

    angular
        .module('annotApp')
        .controller('annotController', annotController);

    annotController.$inject = ['$scope', '$http', '$interval'];
    function annotController($scope, $http, $interval) {
			
		$scope.sent = {};
		
		var selected_indices = {};
		$scope.state = {};	
		$scope.tokens = {};
		$scope.selected = {};
		$scope.current_role = "yellow";
		$scope.f_range_select = false;
		$scope.range_on = false;
		$scope.range_begin = false;
		$scope.range_target_sent = "";
		let range_mode_type = true;
		let cur_range_indices = [];

		function getCookie(cname) {
		  var name = cname + "=";
		  var decodedCookie = decodeURIComponent(document.cookie);
		  var ca = decodedCookie.split(';');
		  for(var i = 0; i <ca.length; i++) {
			var c = ca[i];
			while (c.charAt(0) == ' ') {
			  c = c.substring(1);
			}
			if (c.indexOf(name) == 0) {
			  return c.substring(name.length, c.length);
			}
		  }
		  return "";
		}

		var decodedCookie = decodeURIComponent(document.cookie);
		$scope.seen_ex = (getCookie("seen_ex")=="true");
		console.log($scope.seen_ex);
		function init_state(type, tokens){
			selected_indices[type] = new Set();
			$scope.state[type] = Array(tokens.length);
			$scope.state[type].fill(false);
			$scope.selected[type] = "";
		}

		function init_token(sent_name, sent){
			$scope.sent[sent_name] = sent;
			$scope.tokens[sent_name] = sent.split(" ");
		}

		
		$scope.init = function(){
			console.log($scope.claim1);
			init_token('claim1', $scope.claim1);
			init_token('claim2', $scope.claim2);
			init_state('claim1_yellow', $scope.tokens['claim1']);
			init_state('claim1_red', $scope.tokens['claim1']);
			init_state('claim2_yellow', $scope.tokens['claim2']);
			init_state('claim2_red', $scope.tokens['claim2']);
		}		
		
		$scope.example = function() {
			console.log("click example");
			var d = new Date();
			d.setTime(d.getTime() + (100*24*60*60*1000));
			var expires = "expires="+ d.toUTCString();
			$scope.seen_ex = true;
			var s= "seen_ex=" + $scope.seen_ex +";" + expires + ";path=/";
			document.cookie = s;
		}
		
		$scope.get_role_desc = function(role_name) {
			return {
				'yellow': 'mismatch',
				'red': 'conflict'
			}[role_name]
		}
		
		$scope.next_inst = function(){
			console.log($scope.inst_id);
			
			var next_id = $scope.inst_id + 1
			url = next_id + ".html"
			
		}
		
		
		
		function get_label_type(sent_name, role) {
			return sent_name + "_" + role;
		}
		
		
		$scope.get_visual_type = function(sent_name, index){
			if($scope.state[sent_name + "_yellow"][index])
				var color="yellow";
			else if($scope.state[sent_name + "_red"][index])
				var color="red";
			else
				var color="white";
			return color;
		}
		
		$scope.get_selected = function(sent_name, role){
			return $scope.selected[get_label_type(sent_name, role)];
		}
		
		$scope.get_all_selected = function(){
			return $scope.get_selected('claim1', 'red') + " , "
			+ $scope.get_selected('claim1', 'yellow') + " , "
			+ $scope.get_selected('claim2', 'red') + " , "
			+ $scope.get_selected('claim2', 'yellow');
		}
		
		function click_individual(label_type, index){
			var state = $scope.state[label_type];
			if(!state[index])
				selected_indices[label_type].add(index);
			else
				selected_indices[label_type].delete(index);
			var selected_arr = Array.from(selected_indices[label_type]);
			selected_arr.sort(function(a, b){return a - b});
			$scope.selected[label_type] = selected_arr.join(" ");
			$scope.state[label_type][index] = !$scope.state[label_type][index];
		}

		function click_range(label_type, index){
			if(!$scope.range_on) {
				$scope.range_on = true;
				$scope.range_begin = index;
				$scope.range_target_sent = label_type;
				var state = $scope.state[label_type];
				if (!state[index]) {
					selected_indices[label_type].add(index);
					range_mode_type = true;
				}
				else{
					selected_indices[label_type].delete(index);
					range_mode_type = false;
				}
				$scope.state[label_type][index] = !$scope.state[label_type][index];
			}
			else // assume hover will do job
			{
				console.log("click for off");
				$scope.range_on = false;
				$scope.range_begin = false;
				cur_range_indices = [];
			}
			apply_selected_indices_update(label_type);
		}

		function apply_selected_indices_update(label_type){
			var selected_arr = Array.from(selected_indices[label_type]);
			selected_arr.sort(function(a, b){return a - b});
			$scope.selected[label_type] = selected_arr.join(" ");
		}
		$scope.is_bold = function (sent_name, index){
			let label_type = get_label_type(sent_name, $scope.current_role);
			if($scope.f_range_select
				&& $scope.range_on
				&& $scope.range_begin === index
				&& $scope.range_target_sent === label_type
			)
				return true;
			else
				return false;
		}
		$scope.hover = function(sent_name, index){
			if($scope.f_range_select && $scope.range_on)
			{
				console.log("hover " + sent_name);
				var label_type = get_label_type(sent_name, $scope.current_role);
				if($scope.range_target_sent === label_type){
					var st = Math.min(index, $scope.range_begin);
					var ed = Math.max(index, $scope.range_begin);
					console.log("st, ed", st, ed);
					console.log("cur_range_index", cur_range_indices);
					console.log('label_type', label_type);
					cur_range_indices.forEach( idx => {
						if(range_mode_type){
							$scope.state[label_type][idx] = !range_mode_type;
							console.log('delete', idx, !range_mode_type)
							selected_indices[label_type].delete(idx)
						}
						else{
							$scope.state[label_type][idx] = !range_mode_type;
							console.log('add', idx)
							selected_indices[label_type].add(idx)
						}

					})
					console.log('$scope.state', $scope.state);
					cur_range_indices = [];
					for(let j = st; j <= ed; j++)
					{
						if(range_mode_type){
							$scope.state[label_type][j] = range_mode_type;
							selected_indices[label_type].add(j)
						}
						else{
							$scope.state[label_type][j] = range_mode_type;
							selected_indices[label_type].delete(j)
						}

						cur_range_indices.push(j)
					}
					apply_selected_indices_update(label_type);
				}

			}

		}
		$scope.click = function(sent_name, index){
			console.log(sent_name);
			var label_type = get_label_type(sent_name, $scope.current_role);
			console.log(label_type);
			if($scope.f_range_select)
				click_range(label_type, index)
			else
				click_individual(label_type, index);
		}
		
		$scope.copy = function(label_type){
		  var textArea = document.createElement("textarea");
		  textArea.value = $scope.selected[label_type];
		  document.body.appendChild(textArea);
		  textArea.focus();
		  textArea.select();
		  try {
			var successful = document.execCommand('copy');
			var msg = successful ? 'successful' : 'unsuccessful';
		  } catch (err) {
			console.error('Fallback: Oops, unable to copy', err);
		  }
		  document.body.removeChild(textArea);
		}

	}

})(window.angular);
