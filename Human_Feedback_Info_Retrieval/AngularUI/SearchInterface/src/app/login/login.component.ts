import { HostListener } from '@angular/core';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { ServletService } from '../Service/servlet.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {
  scrHeight:any;
  scrWidth:any;
  user!: string;

  @HostListener('window:resize', ['$event'])
  getScreenSize(event?: undefined) {
        this.scrHeight = window.innerHeight;
        this.scrWidth = window.innerWidth;
        
  }
  constructor(private service : ServletService,
    private router: Router) {  this.getScreenSize();}

  ngOnInit(): void {
  }
  searchUINav():void{
    
    this.service.setData(this.user);
    this.router.navigate([ '/search' ])
    
  }

}
