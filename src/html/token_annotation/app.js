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
		$scope.seen_consent = (getCookie("consent")=="true");
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
			init_token('claim1', $scope.claim1);
			init_token('claim2', $scope.claim2);
			init_state('claim1_yellow', $scope.tokens['claim1']);
			init_state('claim2_yellow', $scope.tokens['claim2']);
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
				'yellow': '',
			}[role_name]
		}
		
		$scope.consent = function() {
			console.log("click consent");
			var d = new Date();
			d.setTime(d.getTime() + (100*24*60*60*1000));
			var expires = "expires="+ d.toUTCString();
			$scope.seen_consent = true;
			var s= "consent=" + $scope.seen_consent + ";"  + expires + ";path=/";
			document.cookie = s;
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
			else
				var color="white";
			return color;
		}
		
		$scope.get_selected = function(sent_name, role){
			return $scope.selected[get_label_type(sent_name, role)];
		}
		
		$scope.get_all_selected = function(){
			return $scope.get_selected('claim1', 'yellow') + " , "
			+ $scope.get_selected('claim2', 'yellow');
		}
		
		
		
		$scope.click = function(sent_name, index){
			var label_type = get_label_type(sent_name, $scope.current_role);
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
		
		$scope.copy = function(){
		  var textArea = document.createElement("textarea");
		  textArea.value = $scope.get_all_selected()
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
